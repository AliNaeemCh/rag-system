from pathlib import Path
import json

def reset_jsonl(output_path: Path):
    """
    Ensures the JSONL file is empty and ready for fresh writing.

    - Creates parent directories if needed
    - Creates the file if it doesn't exist
    - Empties the file if it already exists
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # "w" mode truncates (empties) the file
    with open(output_path, "w", encoding="utf-8"):
        pass

def write_jsonl(records: dict | list[dict], output_path: Path):
    """
    Writes structured records to JSONL.
    """
    if isinstance(records, dict):
        records=[records]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

def get_transcription_by_page(jsonl_path: Path, page_no: int):
    """
    Reads a JSONL file and returns the transcription
    for the given page_no.

    Returns None if page_no is not found.
    """

    if not jsonl_path.exists():
        return None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if record.get("page_no") == page_no:
                return record.get("transcription")

    return None

def extract_completed_page_numbers(jsonl_path: Path) -> list:
    """
    Reads a JSONL file and returns a list of page_no values.
    Skips entries where page_no is missing.
    """

    if not jsonl_path.exists():
        return []

    page_numbers = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if "page_no" in record:
                page_numbers.append(record["page_no"])

    return page_numbers