from dataclasses import dataclass
from typing import Dict, List, Set
from pathlib import Path

@dataclass
class MetadataRule:
    start: int
    end: int
    metadata: Dict


@dataclass
class PDFIngestionConfig:
    pdf_path: Path
    jsonl_path: Path
    start_page: int
    end_page: int
    excluded_pages: Set[int]
    metadata_rules: List[MetadataRule]
    resume_transcription: bool