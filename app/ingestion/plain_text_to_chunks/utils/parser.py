import os
import json
from app.ingestion.plain_text_to_chunks.utils.processors import clean_transcription
from app.core.utils import find_positions
from typing import Iterator
from pathlib import Path

def iter_sections(file_path: Path, h_tags: list[str]) -> Iterator[dict]:
    buffer = ""

    total_size = os.path.getsize(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            transcription = clean_transcription(obj['transcription'])
            page_no = obj['page_no']
            metadata = obj['metadata']
            metadata['page_no'] = page_no

            buffer += transcription

            h_tag_positions = find_positions(text=buffer, pattern=h_tags)
            len_h_tag_positions = len(h_tag_positions)

            if len_h_tag_positions > 0:
                for i in range(len_h_tag_positions - 1):
                    yield {
                        "section": buffer[h_tag_positions[i]: h_tag_positions[i + 1]].strip(" "),
                        "metadata": metadata,
                        "progress": f.tell() / total_size
                    }

                buffer = buffer[h_tag_positions[-1]:]

            else:
                raise Exception(f"<Hx> tag expected in page {page_no} but not found!")

        if buffer:
            yield {
                "section": buffer.strip(" "),
                "progress": 1.0
            }