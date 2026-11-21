"""Fail-closed relationship checks for already security-validated catalog SQL.

Only explicit equality joins on confirmed physical keys are accepted. Direct
column projections through CTEs retain lineage; computed/ambiguous keys do not.
This module never executes model SQL or infers new relationships from it.
"""
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.models.schemas import CatalogProfile


class JoinValidationError(ValueError):
    """A catalog query cannot prove its use of confirmed relationship keys."""


def check_catalog_joins(sql: str, profile: CatalogProfile) -> None:
    tree = sqlglot.parse_one(sql, read="duckdb")
    schemas = {
        d.table_name.casefold(): {c.name.casefold() for c in d.profile.columns}
        for d in profile.datasets
    }
    confirmed = {
        frozenset(((r.left_table.casefold(), r.left_column.casefold()),
                   (r.right_table.casefold(), r.right_column.casefold())))
        for r in profile.relationships if r.status == "confirmed"
    }
    scopes = list(traverse_scope(tree))
    sources_by_scope = {
        id(scope): {alias.casefold(): source for alias, (_, source) in scope.selected_sources.items()}
        for scope in scopes
    }

    def resolve(column, scope, seen=frozenset()):
        if not isinstance(column, exp.Column):
            return None
        token = (id(scope), column.table.casefold(), column.name.casefold())
        if token in seen:
            return None
        seen = seen | {token}
        sources = sources_by_scope[id(scope)]

        def from_source(source):
            if isinstance(source, exp.Table):
                table = source.name.casefold()
                return (table, column.name.casefold()) if column.name.casefold() in schemas.get(table, set()) else None
            if isinstance(source, Scope):
                # An explicit CTE column list renames outputs; reject rather
                # than guessing positional lineage.
                if source.outer_columns:
                    return None
                matches = []
                for projection in source.expression.selects:
                    if projection.alias_or_name.casefold() == column.name.casefold():
                        value = projection.this if isinstance(projection, exp.Alias) else projection
                        matches.append(resolve(value, source, seen))
                return matches[0] if len(matches) == 1 else None
            return None

        if column.table:
            return from_source(sources.get(column.table.casefold()))
        matches = [value for source in sources.values() if (value := from_source(source)) is not None]
        return matches[0] if len(matches) == 1 else None

    physical_tables = {
        source.name.casefold()
        for sources in sources_by_scope.values() for source in sources.values()
        if isinstance(source, exp.Table)
    }
    edges = set()
    for scope in scopes:
        sources = sources_by_scope[id(scope)]
        if scope.subquery_scopes and len(physical_tables) > 1:
            raise JoinValidationError("Cross-dataset analysis requires explicit confirmed joins connecting all referenced datasets; scalar and predicate subqueries are unsupported.")
        if scope.is_correlated_subquery and len(physical_tables) > 1:
            raise JoinValidationError("Cross-dataset correlated subqueries are unsupported. Use an explicit confirmed JOIN with qualified keys.")
        joins = scope.expression.args.get("joins") or []
        if len(sources) > 1 and len(joins) != len(sources) - 1:
            raise JoinValidationError("Each additional dataset must use an explicit confirmed JOIN.")
        joined = set(sources) - {j.this.alias_or_name.casefold() for j in joins}
        for join in joins:
            right_alias = join.this.alias_or_name.casefold()
            condition = join.args.get("on")
            if (join.args.get("method") or join.args.get("using") or condition is None
                    or str(join.args.get("kind", "")).upper() == "CROSS"):
                raise JoinValidationError("Use an explicit JOIN ... ON with confirmed keys. Cross, natural, comma and USING joins are unsupported.")
            predicates = list(condition.flatten()) if isinstance(condition, exp.And) else [condition]
            for predicate in predicates:
                predicate = predicate.unnest()
                if not isinstance(predicate, exp.EQ):
                    raise JoinValidationError("Join conditions must be equality comparisons on confirmed keys. Put other filters in WHERE.")
                left, right = predicate.this, predicate.expression
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column) or not left.table or not right.table:
                    raise JoinValidationError("Join keys must be qualified columns, such as a.customer_id = b.customer_id; computed or ambiguous keys are unsupported.")
                aliases = {left.table.casefold(), right.table.casefold()}
                if right_alias not in aliases or not (aliases & joined) or len(aliases) != 2:
                    raise JoinValidationError("Each JOIN must connect the new dataset to a preceding dataset using its confirmed key.")
                left_key, right_key = resolve(left, scope), resolve(right, scope)
                if left_key is None or right_key is None:
                    raise JoinValidationError("The join key is missing or its source is ambiguous. Use the original columns from the uploaded datasets.")
                if left_key[0] == right_key[0] or frozenset((left_key, right_key)) not in confirmed:
                    raise JoinValidationError("This join relationship is not confirmed. Review the exact columns in Relationships and confirm a valid relationship first.")
                edges.add(frozenset((left_key[0], right_key[0])))
            joined.add(right_alias)
    # Prevent separate scalar/EXISTS subqueries from combining unrelated data
    # without going through an authorized join predicate.
    if physical_tables:
        connected = {next(iter(physical_tables))}
        while True:
            expanded = connected | set().union(*(edge for edge in edges if edge & connected))
            if expanded == connected:
                break
            connected = expanded
        if connected != physical_tables:
            raise JoinValidationError("Cross-dataset analysis requires explicit confirmed joins connecting all referenced datasets.")
