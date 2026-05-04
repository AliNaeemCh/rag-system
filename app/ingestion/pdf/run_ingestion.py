from app.ingestion.pdf.pdf_config_builder import build_pdf_config
from app.ingestion.pdf.pdf_ingestor import PDFIngestor
from app.core.config import settings
from app.infra.clients import gemini_openai_client
from app.infra.llms.engines.openai.engine import OpenAIEngine, OpenAIAPI

raw_config = {
    "pdf_path": settings.RAW_DATA_DIR / "SYS Limited Annual - 2025.pdf",
    "jsonl_path": settings.PROCESSED_DATA_DIR / "SYS Limited Annual - 2025.jsonl",
    "start_page": 5,
    "end_page": 277,
    "exclude_pages": "10,16,28-39,76,80,84,88,93,98,102,123,133-194,273,274",
    "metadata_rules": [
  {
    "start": 5,
    "end": 39,
    "metadata": {
      "chapter": "Company Profile"
    }
  },
  {
    "start": 40,
    "end": 60,
    "metadata": {
      "chapter": "About Systems"
    }
  },
  {
    "start": 61,
    "end": 69,
    "metadata": {
      "chapter": "Human Capital Division"
    }
  },
  {
    "start": 70,
    "end": 75,
    "metadata": {
      "chapter": "Key Financials & Business Highlights"
    }
  },
  {
    "start": 77,
    "end": 101,
    "metadata": {
      "chapter": "Sustainability at a Glance"
    }
  },
  {
    "start": 103,
    "end": 125,
    "metadata": {
      "chapter": "Shareholder's Key Information"
    }
  },
  {
    "start": 126,
    "end": 132,
    "metadata": {
      "chapter": "Corporate Governance"
    }
  },
  {
    "start": 195,
    "end": 272,
    "metadata": {
      "chapter": "Consolidated Financial Statements"
    }
  }
],
    "resume_transcription": True
}

pdf_config = build_pdf_config(raw_config)

pdf_ingestor_llm_engine = OpenAIEngine(gemini_openai_client, settings.PDF_TRANSCRIBER_MODEL, api=OpenAIAPI.CHAT_COMPLETIONS)
pdf_ingestor_fallback_llm_engine = OpenAIEngine(gemini_openai_client, settings.PDF_TRANSCRIBER_FALLBACK_MODEL, api=OpenAIAPI.CHAT_COMPLETIONS)

ingestor = PDFIngestor(llm_engine=pdf_ingestor_llm_engine, fallback_llm_engine=pdf_ingestor_fallback_llm_engine)
ingestor.ingest(pdf_config)