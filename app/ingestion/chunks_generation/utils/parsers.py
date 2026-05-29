from app.ingestion.chunks_generation.utils.processors import clean_transcription
from app.core.utils import find_positions, replace_regex_pattern, combine_dicts, parse_ranges, crossing_index, load_jsonl
from typing import Iterator
from pathlib import Path
import re
from itertools import accumulate

def iter_sections(file_path: Path, h_tags: list[str], last_chunk: dict | None=None) -> Iterator[dict]:
    buffer = []
    metadata = []
    METADATA_COMBINER = " | "

    for line_no, output in enumerate(load_jsonl(file_path, return_progress=True), start=1):
        obj, progress = output

        # <RESUMPTION LOGIC>
        if last_chunk and 'page_no' in obj and last_chunk.get('metadata', {}).get('page_no') is not None:
            last_chunk_page_no = parse_ranges(last_chunk['metadata']['page_no'])[-1]
            last_chunk_content = last_chunk['content']
            if obj['page_no'] == last_chunk_page_no:
                last_chunk_content = replace_regex_pattern(text=last_chunk_content, pattern=re.compile(r"H(\d+):"), replacement=r"<H\1>")    # Hx: -> <Hx>
                transcription = clean_transcription(obj['transcription'])   # Contains document-specific logic
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

        transcription = clean_transcription(obj['transcription'])   # Contains document-specific logic
        page_metadata = obj.get('metadata', {})
        if 'page_no' in obj:
            page_metadata['page_no'] = str(obj['page_no'])
        metadata.append(page_metadata)
        buffer.append(transcription)
        buffer_str = "".join(buffer)
        buffer_chars_count = [len(text) for text in buffer]
        buffer_chars_count_running_sum = list(accumulate(buffer_chars_count))
        h_tag_positions = find_positions(text=buffer_str, pattern=h_tags)
        len_h_tag_positions = len(h_tag_positions)
        if len_h_tag_positions > 0:
            i = 0
            while i < len_h_tag_positions - 1 :
                metadata_idx_start = crossing_index(buffer_chars_count_running_sum, h_tag_positions[i])
                target_pos_for_metadata_idx_end = h_tag_positions[i] + len(buffer_str[h_tag_positions[i]: h_tag_positions[i + 1]].rstrip()) - 1
                metadata_idx_end = crossing_index(buffer_chars_count_running_sum, target_pos_for_metadata_idx_end)
                selected_metadata = combine_dicts([metadata[metadata_idx_start], metadata[metadata_idx_end]], combiner=METADATA_COMBINER)
                if METADATA_COMBINER in selected_metadata.get('page_no', ""):
                    pages_range = parse_ranges(selected_metadata['page_no'])
                    selected_metadata['page_no'] = f"{pages_range[0]}-{pages_range[-1]}"
                yield {
                    "section": buffer_str[h_tag_positions[i]: h_tag_positions[i + 1]].strip(" "),
                    "metadata": selected_metadata,
                    "progress": progress
                }
                i += 1
            if i > 0:
                metadata = [metadata[-1]]

            buffer_str = buffer_str[h_tag_positions[-1]:]
            buffer = [buffer_str]
            if len(metadata) > 1:
                metadata = [combine_dicts(metadata, combiner=METADATA_COMBINER)]
        else:
            raise Exception(f"<Hx> tag expected in line {line_no} but not found!")

    if buffer:
        buffer_str = "".join(buffer)
        yield {
            "section": buffer_str.strip(" "),
            "metadata": metadata[-1],
            "progress": 1.0
        }