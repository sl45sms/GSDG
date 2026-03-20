from typing import Iterable, Tuple


PREFERRED_TEXT_FIELDS = (
    "text",
    "content",
    "document",
    "body",
    "paragraph",
    "sentence",
    "ocr",
    "transcript",
)

SKIP_FIELD_TOKENS = (
    "id",
    "uuid",
    "url",
    "uri",
    "path",
    "link",
    "file",
    "source",
    "date",
    "time",
    "lang",
    "language",
    "label",
    "split",
)


def _normalize_text(value: str, max_chars: int) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _is_informative_string(key: str, value: object) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    lower_key = key.lower()
    if any(token in lower_key for token in SKIP_FIELD_TOKENS):
        return False
    return True


def _iter_string_values(row: dict) -> Iterable[Tuple[str, str]]:
    for key, value in row.items():
        if _is_informative_string(key, value):
            yield key, value


def extract_best_text(row: dict, max_chars: int) -> Tuple[str, str]:
    for field_name in PREFERRED_TEXT_FIELDS:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return field_name, _normalize_text(value, max_chars)

    fragments = []
    fragment_fields = []

    for key, value in _iter_string_values(row):
        fragments.append(value.strip())
        fragment_fields.append(key)

    if not fragments:
        raise ValueError("row does not contain any usable text fields")

    combined = "\n\n".join(fragments)
    return ",".join(fragment_fields), _normalize_text(combined, max_chars)


def infer_row_id(row: dict, row_index: int) -> str:
    for key in ("id", "row_id", "uuid", "doc_id", "document_id"):
        value = row.get(key)
        if value is None:
            continue
        return str(value)
    return str(row_index)
