from app.ingestion.pdf.pdf_config import PDFIngestionConfig, MetadataRule
from app.ingestion.pdf.pdf_utils import parse_ranges


def build_pdf_config(raw: dict) -> PDFIngestionConfig:

    metadata_rules = [
        MetadataRule(
            start=r["start"],
            end=r["end"],
            metadata=r["metadata"]
        )
        for r in raw.get("metadata_rules", [])
    ]

    return PDFIngestionConfig(
        pdf_path=raw["pdf_path"],
        jsonl_path=raw['jsonl_path'],
        start_page=raw["start_page"],
        end_page=raw["end_page"],
        excluded_pages=parse_ranges(raw.get("exclude_pages", "")),
        metadata_rules=metadata_rules,
        resume_transcription=raw['resume_transcription']
    )