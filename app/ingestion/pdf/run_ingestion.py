from app.ingestion.pdf.pdf_config_builder import build_pdf_config
from app.ingestion.pdf.pdf_ingestor import PDFIngestor
from app.core.config import Settings

settings = Settings()

raw_config = {
    "pdf_path": settings.RAW_DATA_DIR / "SYS Limited Annual - 2025.pdf",
    "start_page": 1,
    "end_page": 50,
    "exclude_pages": "1-5,10",
    "metadata_rules": [
        {"start": 1, "end": 3, "metadata": {"section": "intro"}},
        {"start": 4, "end": 10, "metadata": {"section": "financials"}}
    ]
}

config = build_pdf_config(raw_config)

ingestor = PDFIngestor(llm_engine, extractor, writer)

ingestor.ingest(config)