import re
from pathlib import Path
import json
from collections import defaultdict

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

    values = set()

    if not range_str or not range_str.strip():
        return values

    for part in re.split(r"[,\|]", range_str):
        part = part.strip()

        if not part:
            raise ValueError("Empty segment in range string.")

        # Match ranges like 1-3
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)

        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))

            # enforce ascending order
            if start > end:
                raise ValueError(
                    f"Invalid range '{part}': start must be <= end."
                )

            values.update(range(start, end + 1))
            continue

        # Match single number
        if part.isdigit():
            values.add(int(part))
            continue

        # Anything else is invalid
        raise ValueError(f"Invalid range segment: '{part}'")

    return list(values)

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