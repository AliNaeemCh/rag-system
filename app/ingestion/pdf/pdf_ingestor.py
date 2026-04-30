import json
import logging
from pathlib import Path
from app.ingestion.pdf.pdf_utils import pdf_page_to_image, get_pdf_page_count

logger = logging.getLogger("app.ingestion.pdf.pdf_ingestor")


class PDFIngestor:
    """
    Pipeline:
        PDF -> page transcription -> JSONL
    """

    def __init__(self, llm_engine):
        self.llm_engine = llm_engine

    # ---------------- PAGE TRANSCRIPTION ----------------
    def transcribe_page(self, page_content: str) -> str:
        """
        Uses LLM to clean / transcribe page content.
        """

        prompt = f"""
Transcribe and clean the following PDF page content.

Preserve:
- tables
- headings
- numbers
- financial values

Do not summarize.

CONTENT:
{page_content}
"""

        response = self.llm_engine.generate(
            input_messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.strip()

    # ---------------- MAIN PIPELINE ----------------
    def ingest(
        self,
        config,
        output_jsonl_path: str
    ):
        """
        Full ingestion pipeline.
        """

        logger.info(f"Ingestion started | pdf={config.pdf_path}")
        records = []

        for page in iter_pages(config):

            raw_text = self.extractor.extract(config.pdf_path, page)

            cleaned = self.llm_engine.transcribe(raw_text)

            metadata = get_metadata(page, config.metadata_rules)

            records.append({
                "page": page,
                "text": cleaned,
                "metadata": metadata,
                "source": config.pdf_path
            })

        logger.info(
            f"Ingestion completed | pdf={pdf_path}"
        )

import logging
from app.ingestion.page_selector import iter_pages
from app.ingestion.metadata import get_metadata

logger = logging.getLogger("ingestor")


class PDFIngestor:

    def __init__(self, llm_engine, extractor, writer):
        self.llm_engine = llm_engine
        self.extractor = extractor
        self.writer = writer

    def ingest(self, config):

        records = []

        for page in iter_pages(config):

            raw_text = self.extractor.extract(config.pdf_path, page)

            cleaned = self.llm_engine.transcribe(raw_text)

            metadata = get_metadata(page, config.metadata_rules)

            records.append({
                "page": page,
                "text": cleaned,
                "metadata": metadata,
                "source": config.pdf_path
            })

        self.writer.write(records)