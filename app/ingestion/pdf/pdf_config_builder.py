from app.ingestion.pdf.pdf_config import IngestionConfig, MetadataRule
from app.ingestion.pdf.page_selector import parse_exclusions


def build_pdf_config(raw: dict) -> IngestionConfig:
    metadata_rules = [
        MetadataRule(
            start=r["start"],
            end=r["end"],
            metadata=r["metadata"]
        )
        for r in raw.get("metadata_rules", [])
    ]

    return IngestionConfig(
        pdf_path=raw["pdf_path"],
        start_page=raw["start_page"],
        end_page=raw["end_page"],
        excluded_pages=parse_exclusions(raw.get("exclude_pages", "")),
        metadata_rules=metadata_rules
    )