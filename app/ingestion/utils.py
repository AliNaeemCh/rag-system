from pathlib import Path
import json

def write_jsonl(records: list[dict], output_path: str):
    """
    Writes structured records to JSONL.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")