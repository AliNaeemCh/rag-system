import re
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