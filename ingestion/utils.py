from app.core.utils import get_jsonl_object

import logging
logger = logging.getLogger("ingestion.utils")
logger.info("Loading file...")

from pathlib import Path
import json

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

def get_chunk_obj(chunks_path: Path, chunks_index: list[int], chunk_id: int) -> dict:
    chunk_obj = get_jsonl_object(jsonl_path=chunks_path, index=chunks_index, line_number=chunk_id)
    if chunk_obj['chunk_id'] != chunk_id:
        raise Exception(f"Mismatch in chunks file and index. Extracted chunk id={chunk_id} and got id={chunk_obj['chunk_id']}")
    
    return chunk_obj