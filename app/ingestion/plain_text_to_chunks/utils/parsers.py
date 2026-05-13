import os
import json
from app.ingestion.plain_text_to_chunks.utils.processors import clean_transcription
from app.core.utils import find_positions, replace_regex_pattern, combine_dicts, parse_ranges
from typing import Iterator
from pathlib import Path
import re
from itertools import accumulate

def crossing_index(running_sums, target):
    """
    running_sums: list of cumulative sums
    target: number to check

    returns index where running sum first crosses target
    """

    for i, value in enumerate(running_sums):
        if value > target:
            return i

    return -1

def iter_sections(file_path: Path, h_tags: list[str], last_chunk: dict | None=None) -> Iterator[dict]:
    buffer = []
    metadata = []
    processed_bytes = 0
    total_size = os.path.getsize(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # <RESUMPTION LOGIC>
            if last_chunk:
                last_chunk_page_no = parse_ranges(last_chunk['metadata']['page_no'])[-1]    # TO CHECK
                last_chunk_content = last_chunk['content']
                if obj['page_no'] == last_chunk_page_no:
                    last_chunk_content = replace_regex_pattern(text=last_chunk_content, pattern=re.compile(r"H(\d+):"), replacement=r"<H\1>")    # Hx: -> <Hx>
                    transcription = clean_transcription(obj['transcription'])
                    h_pos = find_positions(transcription, pattern=re.compile(r"<H\d+>"))
                    if h_pos:
                        len_h_tag = 4   # <Hx> length
                        last_h_pos = h_pos[-1]
                        last_h_level = transcription[last_h_pos: last_h_pos+len_h_tag]
                        if last_h_level == '<H1>':
                            h1_section_name_end_pos = transcription.find('\n', last_h_pos)
                            h1_section_name = transcription[last_h_pos:h1_section_name_end_pos]
                            if h1_section_name not in last_chunk_content:
                                buffer = [transcription[last_h_pos:]]
                                metadata = [last_chunk['metadata']]
                    last_chunk = None  
                continue
            # <RESUMPTION LOGIC>

            transcription = clean_transcription(obj['transcription'])
            page_metadata = obj['metadata']
            if 'page_no' in obj:
                page_metadata['page_no'] = obj['page_no']
            metadata.append(page_metadata)
            buffer.append(transcription)
            buffer_str = "".join(buffer)
            buffer_chars_count = [len(text) for text in buffer]
            buffer_chars_count_running_sum = list(accumulate(buffer_chars_count))
            h_tag_positions = find_positions(text=buffer_str, pattern=h_tags)
            len_h_tag_positions = len(h_tag_positions)
            processed_bytes += len(line.encode("utf-8")) + 1
            if len_h_tag_positions > 0:
                for i in range(len_h_tag_positions - 1):
                    metadata_idx = crossing_index(buffer_chars_count_running_sum, h_tag_positions[i])
                    yield {
                        "section": buffer_str[h_tag_positions[i]: h_tag_positions[i + 1]].strip(" "),
                        "metadata": metadata[metadata_idx],
                        "progress": processed_bytes / total_size
                    }

                buffer_str = buffer_str[h_tag_positions[-1]:]
                buffer = [buffer_str]
                metadata = [metadata[-1]]
            else:
                raise Exception(f"<Hx> tag expected in line {line_no} but not found!")

        if buffer:
            buffer_str = "".join(buffer)
            yield {
                "section": buffer_str.strip(" "),
                "progress": 1.0
            }

def extract_last_chunk(jsonl_path: Path) -> int | None:
    """
    Reads only the last JSONL record and returns metadata.page_no.
    """

    if not jsonl_path.exists():
        return None

    with open(jsonl_path, "rb") as f:
        f.seek(0, 2)  # move to end of file

        if f.tell() == 0:
            return None

        buffer = bytearray()

        pointer = f.tell() - 1

        while pointer >= 0:
            f.seek(pointer)
            byte = f.read(1)

            if byte == b"\n" and buffer:
                break

            buffer.extend(byte)
            pointer -= 1

        last_line = buffer[::-1].decode("utf-8").strip()

    if not last_line:
        return None

    record = json.loads(last_line)

    return record
