from dataclasses import dataclass
from pathlib import Path

@dataclass
class MetadataRule:
    start: int
    end: int
    metadata: dict


@dataclass
class PDFIngestionConfig:
    pdf_path: Path
    jsonl_path: Path
    start_page: int
    end_page: int
    excluded_pages: list[int]
    metadata_rules: list[MetadataRule]
    resume_transcription: bool