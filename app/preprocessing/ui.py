"""Optional Streamlit preprocessing controls and change summary."""
import streamlit as st

from app.preprocessing.change_report import build_change_report
from app.preprocessing.processor import OPERATIONS, process_dataframe


def render_report(report: dict) -> None:
    st.subheader("Preprocessing change report")
    st.caption(f"Original: {report['original_table']} · Processed: {report['processed_table']}")
    before, after = report["score_before"], report["score_after"]
    first, second, third = st.columns(3)
    first.metric("Quality before", f"{before['Overall']:.1f}/100")
    second.metric("Quality after", f"{after['Overall']:.1f}/100")
    third.metric("Difference", f"{report['score_difference']:+.1f}")
    st.dataframe([{"Dimension": name, "Before": round(before[name], 1),
                   "After": round(after[name], 1)} for name in before if name != "Overall"],
                 hide_index=True, use_container_width=True)
    with st.expander("How this is calculated"):
        st.write("Completeness counts nonblank cells. Uniqueness counts distinct rows. Type consistency counts text columns that do not need safe conversion. Text consistency counts trimmed text values. Schema cleanliness counts unique lowercase snake_case names. Overall is the average of these five scores.")
        st.caption("This measures structural cleanliness, not real-world accuracy.")
    st.write("Selected:", ", ".join(report["selected_operations"]))
    st.write("Completed:", ", ".join(report["completed"]) or "None")
    st.write("Skipped:", report["skipped"] or "None")
    st.write(f"Rows: {report['original_rows']} → {report['processed_rows']} · Columns: {report['original_columns']} → {report['processed_columns']}")
    st.write(f"Exact duplicates removed: {report['duplicates_removed']} · Missing cells: {report['missing_before']} → {report['missing_after']}")
    st.write(f"Text values changed: {report['text_values_changed']}")
    st.write("Converted columns:", report["converted_columns"] or "None")
    st.write("Renamed columns:", report["renamed_columns"] or "None")
    left, right = st.columns(2)
    left.caption("Before preview")
    left.dataframe(report["before_preview"], hide_index=True, use_container_width=True)
    right.caption("After preview")
    right.dataframe(report["after_preview"], hide_index=True, use_container_width=True)
    frame = st.session_state.dataset_catalog.read_table(report["processed_id"])
    st.download_button("Download Cleaned Dataset", frame.to_csv(index=False).encode("utf-8"),
                       file_name=f"{report['processed_table']}.csv", mime="text/csv")


def _submit(selected, catalog, selected_id, prepare_catalog, refresh_catalog, activate_catalog):
    if not selected:
        st.info("Select at least one preprocessing operation.")
        return
    try:
        if catalog is None:
            catalog, selected_id = prepare_catalog()
        original = catalog.read_table(selected_id)
        processed, details = process_dataframe(original, selected)
        report = build_change_report(original, processed, details)
        record = catalog.register_processed(selected_id, processed, report)
        report.update(original_table=next(item.table_name for item in catalog.profile.datasets
                                          if item.dataset_id == selected_id),
                      processed_table=record.table_name, processed_id=record.dataset_id)
        if st.session_state.dataset_catalog is None:
            activate_catalog(catalog, st.session_state.documents,
                             st.session_state.uploaded_file_signature,
                             st.session_state.pdf_page_errors, st.session_state.pdf_notices)
        else:
            refresh_catalog()
        st.session_state.preprocess_report = report
        st.session_state.show_preprocess = True
        st.success(f"Created {record.table_name}; the original table is unchanged.")
        st.rerun()
    except Exception as exc:
        if catalog is not st.session_state.dataset_catalog and catalog is not None:
            catalog.close()
        st.error(f"Could not preprocess this dataset: {exc}")


def render_preprocessing(prepare_catalog, refresh_catalog, activate_catalog) -> None:
    if st.session_state.dataframe is None or st.session_state.upload_error:
        return
    if st.button("Data Preprocess"):
        st.session_state.show_preprocess = True
    if not st.session_state.get("show_preprocess"):
        return
    catalog = st.session_state.dataset_catalog
    originals = ([record for record in catalog.profile.datasets
                  if not record.table_name.endswith("_processed")] if catalog else [])
    selected_id = None
    if originals:
        selected_id = st.selectbox("Dataset to preprocess", [record.dataset_id for record in originals],
                                   format_func=lambda item: next(record.filename for record in originals
                                                                 if record.dataset_id == item))
    selected = st.multiselect("Preprocessing operations", list(OPERATIONS))
    if st.button("Submit", key="submit_preprocessing"):
        _submit(selected, catalog, selected_id, prepare_catalog, refresh_catalog, activate_catalog)
    if st.session_state.get("preprocess_report"):
        render_report(st.session_state.preprocess_report)
