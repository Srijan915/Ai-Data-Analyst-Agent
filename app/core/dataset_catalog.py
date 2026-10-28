"""Session-private, bounded multi-CSV catalog backed by one DuckDB database.

Uploaded CSV bytes are first parsed by :class:`DiskDataset`, then copied with
trusted SQL into an internal catalog database.  Model-generated SQL can only
see the materialized, allowlisted tables through the normal validator.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import threading
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from app.config import config
from app.core.disk_dataset import DiskDataset, _BoundedConnection, disk_budget
from app.core.sql_executor import SQLExecutor
from app.models.schemas import CatalogProfile, DatasetMetadata, RelationshipSuggestion
from app.utils.dataset_resources import DatasetResourceError
from app.utils.temporary_storage import runtime_directory


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _safe_table_name(filename: str, digest: str, occupied: set[str]) -> str:
    stem = re.sub(r"[^a-z0-9_]+", "_", Path(filename).stem.lower()).strip("_")
    stem = stem[:36] or "upload"
    base = f"ds_{stem}_{digest[:10]}"
    candidate, suffix = base, 2
    while candidate in occupied:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


class DatasetCatalog:
    """Own multiple datasets for exactly one browser/session lifecycle."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="catalog-", dir=runtime_directory()))
        self.path = self.root / "catalog.duckdb"
        self._records: dict[str, DatasetMetadata] = {}
        self._lock = threading.RLock()
        self._users = 0
        self._closed = False
        self._reservations: dict[str, object] = {}
        self._relationship_status: dict[tuple[str, str, str, str], str] = {}
        self._finalizer = weakref.finalize(self, shutil.rmtree, self.root, True)

    @property
    def profile(self) -> CatalogProfile:
        return CatalogProfile(datasets=list(self._records.values()), relationships=self._relationships())

    def _relationships(self) -> list[RelationshipSuggestion]:
        """Suggest only name/type-compatible keys; suggestions never authorize joins."""
        records = list(self._records.values())
        suggestions = []
        for index, left in enumerate(records):
            for right in records[index + 1:]:
                for left_column in left.profile.columns:
                    for right_column in right.profile.columns:
                        if left_column.name.casefold() != right_column.name.casefold():
                            continue
                        if left_column.inferred_type != right_column.inferred_type:
                            continue
                        overlap = self._value_overlap(
                            left.table_name, left_column.name,
                            right.table_name, right_column.name,
                        )
                        # Shared names alone are not a relationship. Avoid
                        # surfacing suggestions with no observed value overlap.
                        if overlap == 0:
                            continue
                        confidence = "high" if left_column.unique_count == left.row_count or right_column.unique_count == right.row_count else "medium"
                        key = (left.table_name, left_column.name, right.table_name, right_column.name)
                        suggestions.append(RelationshipSuggestion(
                            left_table=left.table_name, left_column=left_column.name,
                            right_table=right.table_name, right_column=right_column.name,
                            confidence=confidence, overlap_percentage=overlap,
                            status=self._relationship_status.get(key, "suggested"),
                        ))
        return suggestions

    def _value_overlap(self, left_table: str, left_column: str, right_table: str, right_column: str) -> float:
        """Bounded observed overlap for a suggestion, never planner input alone."""
        if not self.path.exists():
            return 0.0
        conn = duckdb.connect(str(self.path), read_only=True)
        try:
            left = {row[0] for row in conn.execute(
                f"SELECT DISTINCT {_quote(left_column)} FROM {_quote(left_table)} WHERE {_quote(left_column)} IS NOT NULL LIMIT 10001"
            ).fetchall()}
            right = {row[0] for row in conn.execute(
                f"SELECT DISTINCT {_quote(right_column)} FROM {_quote(right_table)} WHERE {_quote(right_column)} IS NOT NULL LIMIT 10001"
            ).fetchall()}
        finally:
            conn.close()
        if not left or not right:
            return 0.0
        return round(100 * len(left & right) / min(len(left), len(right)), 2)

    def set_relationship_status(
        self, left_table: str, left_column: str, right_table: str,
        right_column: str, status: str,
    ) -> None:
        """Explicit user approval/rejection is required before joins are planned."""
        if status not in {"confirmed", "rejected"}:
            raise ValueError("relationship status must be confirmed or rejected")
        key = (left_table, left_column, right_table, right_column)
        if not any((r.left_table, r.left_column, r.right_table, r.right_column) == key for r in self._relationships()):
            raise ValueError("unknown relationship suggestion")
        self._relationship_status[key] = status

    def add(self, upload, filename: str | None = None, progress=None) -> DatasetMetadata:
        """Validate and materialize an upload without retaining its raw bytes."""
        if self._closed:
            raise DatasetResourceError("Dataset session has been cleared.")
        source = DiskDataset(upload, progress=progress)
        reservation = None
        try:
            digest = hashlib.sha256(source.raw.read_bytes()).hexdigest()
            name = filename or getattr(upload, "name", "upload.csv")
            if not str(name).lower().endswith(".csv"):
                raise DatasetResourceError("Only CSV uploads are supported.")
            with self._lock:
                table_name = _safe_table_name(str(name), digest, set(self.profile.table_names))
                # Reserve catalog capacity before copying; the source reservation
                # remains live until this succeeds, so the global limit is real.
                # The parser already reserved its maximum while validating the
                # source.  Reserve a conservative, content-based amount for
                # the materialized copy so several normal-size CSVs can share
                # the session budget (rather than one upload consuming the
                # entire configured maximum forever).
                copy_bytes = max(1024**2, min(
                    config.max_dataset_disk_mb * 1024**2,
                    source.raw.stat().st_size * 3,
                ))
                reservation = disk_budget.reserve(copy_bytes)
                conn = duckdb.connect(str(self.path))
                try:
                    conn.execute(f"ATTACH {_sql_literal(str(source.path))} AS source_db (READ_ONLY)")
                    conn.execute(f"CREATE TABLE {_quote(table_name)} AS SELECT * FROM source_db.dataset")
                    conn.execute("DETACH source_db")
                    conn.execute(f"SET memory_limit='{config.duckdb_memory_limit}'")
                    conn.execute(f"SET threads={config.duckdb_thread_limit}")
                    conn.execute("SET temp_directory=''")
                    conn.execute("SET enable_external_access=false")
                finally:
                    conn.close()
                record = DatasetMetadata(
                    dataset_id=str(uuid.uuid4()), filename=Path(str(name)).name,
                    table_name=table_name,
                    uploaded_at=datetime.now(timezone.utc).isoformat(),
                    row_count=source.row_count, profile=source.execution_profile,
                )
                self._records[record.dataset_id] = record
                self._reservations[record.dataset_id] = reservation
                reservation = None
                return record
        except Exception:
            if reservation is not None:
                reservation.release()
            raise
        finally:
            source.close()

    def remove(self, dataset_id: str) -> None:
        with self._lock:
            record = self._records.pop(dataset_id, None)
            if record is None:
                return
            conn = duckdb.connect(str(self.path))
            try:
                conn.execute(f"DROP TABLE {_quote(record.table_name)}")
            finally:
                conn.close()
            reservation = self._reservations.pop(dataset_id, None)
            if reservation is not None:
                reservation.release()

    def replace(self, dataset_id: str, upload, filename: str | None = None, progress=None) -> DatasetMetadata:
        """Replace one table atomically from the caller's perspective.

        The new CSV is fully parsed/materialized first.  If that fails, the
        original record and table remain available.  On success its inferred
        relationships disappear naturally when the old table is removed.
        """
        if dataset_id not in self._records:
            raise ValueError("unknown dataset")
        replacement = self.add(upload, filename=filename, progress=progress)
        self.remove(dataset_id)
        return replacement

    def acquire(self) -> None:
        with self._lock:
            if self._closed:
                raise DatasetResourceError("Dataset session has been cleared.")
            self._users += 1

    def release(self) -> None:
        with self._lock:
            self._users -= 1
            if self._closed and not self._users:
                self._destroy()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if not self._users:
                self._destroy()

    def _destroy(self) -> None:
        if self._finalizer.alive:
            self._finalizer()
        for reservation in self._reservations.values():
            reservation.release()
        self._reservations.clear()


class CatalogSQLExecutor(SQLExecutor):
    """Read-only executor for a session catalog; raw SQL remains forbidden."""

    def __init__(self, catalog: DatasetCatalog, max_result_rows: int | None = None):
        self._catalog = catalog
        self._table_names = catalog.profile.table_names
        self._table_name = self._table_names[0] if len(self._table_names) == 1 else None
        self._max_result_rows = max_result_rows or config.max_query_result_rows
        self._timeout_seconds = config.query_timeout_seconds
        self._timeout_cleanup_pending = False
        self._released = False
        catalog.acquire()
        try:
            conn = duckdb.connect(str(catalog.path), read_only=True)
            conn.execute(f"SET memory_limit='{config.duckdb_memory_limit}'")
            conn.execute(f"SET threads={config.duckdb_thread_limit}")
            conn.execute("SET temp_directory=''")
            conn.execute("SET enable_external_access=false")
            self._conn = _BoundedConnection(conn, self._max_result_rows)
        except Exception:
            catalog.release()
            raise

    @property
    def table_names(self) -> list[str]:
        return list(self._table_names)

    def _release_catalog(self) -> None:
        if not self._released:
            self._released = True
            self._catalog.release()

    def close(self) -> None:
        if not self._timeout_cleanup_pending and not self._released:
            try:
                super().close()
            finally:
                self._release_catalog()

    def _close_after_worker_finishes(self, future) -> None:
        try:
            super()._close_after_worker_finishes(future)
        finally:
            self._release_catalog()
