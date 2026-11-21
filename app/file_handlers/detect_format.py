"""Bounded upload reading and signature-based format detection."""
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from app.config import config


def read_upload(upload) -> bytes:
    name = getattr(upload, "name", "uploaded file")
    limit = config.max_upload_size_mb * 1024 * 1024
    if getattr(upload, "size", 0) > limit:
        raise ValueError(f"{name} exceeds the {config.max_upload_size_mb} MB upload limit.")
    try:
        if hasattr(upload, "getvalue"):
            data = upload.getvalue()
        else:
            position = upload.tell() if hasattr(upload, "tell") else None
            upload.seek(0)
            data = upload.read(limit + 1)
            if position is not None:
                upload.seek(position)
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError(f"Could not read {name}.") from exc
    if not isinstance(data, bytes) or not data:
        raise ValueError(f"{name} is empty or unreadable.")
    if len(data) > limit:
        raise ValueError(f"{name} exceeds the {config.max_upload_size_mb} MB upload limit.")
    return data


def detect_format(name: str, data: bytes) -> tuple[str, str | None]:
    if not data:
        raise ValueError(f"{name} is empty.")
    extension = Path(name).suffix.lower()
    if extension not in {".csv", ".xlsx", ".txt", ".pdf"}:
        raise ValueError(f"Unsupported file type {extension or '(none)'}. Use CSV, XLSX, TXT or PDF.")
    if data.startswith(b"%PDF"):
        kind = "pdf"
    elif data.startswith(b"PK\x03\x04"):
        try:
            with ZipFile(BytesIO(data)) as archive:
                kind = "xlsx" if "xl/workbook.xml" in archive.namelist() else "unsupported"
                if sum(item.file_size for item in archive.infolist()) > 2 * config.max_dataset_memory_mb * 1024**2:
                    raise ValueError("XLSX contents exceed the dataset memory budget.")
        except BadZipFile as exc:
            raise ValueError(f"{name} is an unreadable ZIP/XLSX file.") from exc
    elif any(byte == 0 or (byte < 32 and byte not in (9, 10, 13)) for byte in data[:4096]):
        kind = "unsupported"
    else:
        kind = "csv" if extension == ".csv" else "txt"
    if kind == "unsupported":
        raise ValueError(f"{name} is not a supported CSV, XLSX, TXT or PDF file.")
    expected = "txt" if extension in {".txt", ".xlsx", ".pdf"} and kind in {"csv", "txt"} else extension.lstrip(".")
    warning = (f"{name}: extension {extension} differs from detected {kind.upper()} content; "
               f"using {kind.upper()}.") if expected != kind or (kind == "txt" and extension != ".txt") else None
    return kind, warning
