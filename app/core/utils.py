import logging
logger = logging.getLogger("app.core.utils")
logger.info("Loading file...")

import re
from pathlib import Path
import json
from collections import defaultdict
from typing import Iterator
import os

def reset_jsonl(jsonl_path: Path):
    """
    Ensures the JSONL file is empty and ready for fresh writing.

    - Creates parent directories if needed
    - Creates the file if it doesn't exist
    - Empties the file if it already exists
    """

    jsonl_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # "w" mode truncates (empties) the file
    with open(jsonl_path, "w", encoding="utf-8"):
        pass

def write_jsonl(data: dict | list[dict], output_path: Path):
    """
    Writes structured records to JSONL.
    """
    if isinstance(data, dict):
        data=[data]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    lines = "\n".join(
        json.dumps(data_item, ensure_ascii=False)
        for data_item in data
    ) + "\n"

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(lines)

def load_jsonl(path: Path, return_progress=False) -> Iterator[dict | tuple[dict, float]]:
    """
    Lazy streaming JSONL reader (1 line at a time)
    If `return_progress` is set True, it returns the progress which is a float in range 0 to 1
    """
    total_size = os.path.getsize(path)
    processed_bytes = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            processed_bytes += len(line.encode("utf-8")) + 1
            progress = processed_bytes / total_size
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if return_progress:
                    yield obj, progress
                else:
                    yield obj
            except json.JSONDecodeError:
                continue

def write_json(data: dict | list[dict], output_path: Path):
    """
    Writes structured records to a JSON file (not JSONL).
    """

    # Normalize to list
    if isinstance(data, dict):
        data = [data]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

def reset_json(json_path: Path):
    """
    Ensures the JSON file is initialized and ready for fresh writing.

    - Creates parent directories if needed
    - Creates the file if it doesn't exist
    - Resets it to an empty JSON array
    """

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([], f)

def find_positions(text: str, pattern: str | list[str] | re.Pattern, find_first=False) -> list[int] | int | None:
    """
    Returns start positions of all matches of a string, list of strings, or regex pattern.
    
    Args:
        text: input text
        pattern: string, list of strings, or compiled regex re.compile(...)
        find_first: True will return the position number of first match or None. False will return list of positions.
    """

    if isinstance(pattern, list):
        # escape each string and join with OR
        pattern = "|".join(re.escape(p) for p in pattern)
    elif isinstance(pattern, str):
        pattern = re.escape(pattern)

    if find_first:
        match = re.search(pattern, text)
        return match.start() if match else None

    return [m.start() for m in re.finditer(pattern, text)]

def replace_regex_pattern(text: str, pattern: re.Pattern, replacement: str):
    def repl(match: re.Match):
        # replace \1, \2, etc. in replacement string
        out = replacement
        for i, group in enumerate(match.groups(), start=1):
            out = out.replace(f"\\{i}", group)
        return out

    return pattern.sub(repl, text)

def parse_ranges(range_str: str) -> list[int]:
    """
    Parses range strings like:
        "1-3,5,6|9"

    Rules:
        - ranges must be ascending (1-3 valid, 3-1 invalid)
        - invalid input raises ValueError
        - whitespace is allowed

    Examples:
        "1-3,5,6|9" -> {1,2,3,5,6,9}
        "2,4-6"   -> {2,4,5,6}
    """

    if not range_str or not range_str.strip():
        return []

    values = []
    seen = set()

    def add(x: int):
        if x not in seen:
            seen.add(x)
            values.append(x)

    for part in re.split(r"[,\|]", range_str):
        part = part.strip()

        if not part:
            raise ValueError("Empty segment in range string.")

        # Match range like 1-3
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)

        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))

            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end.")

            for i in range(start, end + 1):
                add(i)

            continue

        # Single number
        if part.isdigit():
            add(int(part))
            continue

        raise ValueError(f"Invalid range segment: '{part}'")

    return values

def combine_dicts(dict_list: list[dict], combiner: str = " | ") -> dict:
    """
    Combine a list of dictionaries.

    Rules:
    - If a key appears only once, keep its value as-is.
    - If a key appears in multiple dicts:
        * convert all values to str
        * combine unique values with `combiner`

    Example: (`combiner=', '`)
    [
        {"a": 1, "b": "x"},
        {"a": 2, "c": True},
        {"b": "y", "a": 1}
    ]

    =>
    {
        "a": "1, 2",
        "b": "x, y",
        "c": True
    }
    """

    values = defaultdict(list)
    counts = defaultdict(int)

    # Collect values
    for d in dict_list:
        for k, v in d.items():
            counts[k] += 1
            values[k].append(v)

    result = {}

    for k, vals in values.items():
        if counts[k] == 1:
            # Keep original value if key appeared once
            result[k] = vals[0]
        else:
            # Combine unique stringified values
            seen = []
            for v in vals:
                sv = str(v)
                if sv not in seen:
                    seen.append(sv)

            result[k] = combiner.join(seen)

    return result

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

def extract_last_jsonl_object(jsonl_path: Path) -> dict:
    """
    Returns only the last JSONL object
    """

    if not jsonl_path.exists():
        return {}

    with open(jsonl_path, "rb") as f:
        f.seek(0, 2)  # move to end of file

        if f.tell() == 0:
            return {}

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
        return {}

    object = json.loads(last_line)

    return object