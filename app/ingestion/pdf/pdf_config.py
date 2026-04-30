from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class MetadataRule:
    start: int
    end: int
    metadata: Dict


@dataclass
class IngestionConfig:
    pdf_path: str
    start_page: int
    end_page: int
    excluded_pages: Set[int]
    metadata_rules: List[MetadataRule]