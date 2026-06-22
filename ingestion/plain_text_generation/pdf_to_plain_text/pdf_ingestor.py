from ingestion.plain_text_generation.pdf_to_plain_text.utils import pdf_page_to_image
from ingestion.utils import get_transcription_by_page, extract_completed_page_numbers
from app.core.utils import write_jsonl, reset_jsonl
from ingestion.plain_text_generation.pdf_to_plain_text.config.pdf_config import PDFIngestionConfig
from app.infra.llm_engines.base import BaseLLMEngine
from ingestion.plain_text_generation.pdf_to_plain_text.utils import get_metadata
from app.prompts.pdf_transcriber import PDF_TRANSCRIBER_SYSTEM_PROMPT

import logging
logger = logging.getLogger("ingestion.plain_text_generation.pdf_to_plain_text.pdf_ingestor")
logger.info("Loading file...")

import re
from tqdm import tqdm

class PDFIngestor:
    """
    Pipeline:
        PDF -> page transcription -> JSONL
    """

    def __init__(self, llm_engine: BaseLLMEngine, fallback_llm_engine: BaseLLMEngine | None = None):
        self.llm_engine = llm_engine
        self.fallback_llm_engine = fallback_llm_engine

    def _extract_partial_content(self, text: str) -> str:
        """
        Extract:
        - the LAST <H1>
        - all following headings (<H2>, <H3>, ...)

        For all headings except the last one → use "..."
        For the last heading → include full remaining content
        """

        output = []

        pattern = re.compile(
            r"<H(\d)>\s*(.*?)(?:\s*</H\1>|$|\n)",
            re.IGNORECASE
        )

        matches = list(pattern.finditer(text))

        # Find last H1
        last_h1_index = -1

        for i, match in enumerate(matches):
            if match.group(1) == "1":
                last_h1_index = i

        if last_h1_index == -1:
            return ""

        selected = matches[last_h1_index:]

        for i, match in enumerate(selected):
            level = match.group(1)
            title = match.group(2).strip()

            title = title.split("\n")[0].strip()
            title = re.sub(r"\s+", " ", title)

            if not title:
                continue

            # Last section gets content instead of "..."
            if i == len(selected) - 1:
                # extract content until next heading or end
                start = match.end()
                next_match = selected[i + 1] if i + 1 < len(selected) else None
                end = next_match.start() if next_match else len(text)

                content = text[start:end].strip()

                output.append(f"<H{level}> {title}\n{content}")
            else:
                output.append(f"<H{level}> {title}\n...")

        return "\n".join(output)
    
    def _get_pages_info(self, pdf_config: PDFIngestionConfig):
        pages_completed = []
        if pdf_config.resume_transcription:
            pages_completed = extract_completed_page_numbers(pdf_config.jsonl_path)
        pages_list = []
        for page_no in range(pdf_config.start_page, pdf_config.end_page + 1):
            if page_no not in pdf_config.excluded_pages and page_no not in pages_completed:
                pages_list.append(page_no)
        total_pages = (pdf_config.end_page - pdf_config.start_page + 1) - len(pdf_config.excluded_pages)
        return pages_list, pages_completed, total_pages

    # ---------------- MAIN PIPELINE ----------------
    def ingest(
        self,
        pdf_config: PDFIngestionConfig
    ):
        """
        Full ingestion pipeline.
        """
        pages_list, pages_completed, total_pages = self._get_pages_info(pdf_config)
        if not pdf_config.resume_transcription:
            logger.info(f"Ingestion started | pdf={pdf_config.pdf_path}")
            reset_jsonl(pdf_config.jsonl_path)
        else:
            logger.info(f"Ingestion resumed | pdf={pdf_config.pdf_path}")
        if len(pages_list) > 0 and pages_list[0] - 1 in pages_completed:
            # Getting previous section content structure
            page_no = pages_list[0] - 1
            previous_pages_transcripts = []
            while True:
                page_transcript = get_transcription_by_page(pdf_config.jsonl_path, page_no=page_no)
                previous_pages_transcripts.insert(0, page_transcript)
                if '<CONT.>' not in page_transcript:
                    break
                page_no -= 1
                if page_no not in pages_completed:
                    break
            previous_pages_transcription = "\n\n".join(previous_pages_transcripts)
            previous_section_transcription = self._extract_partial_content(previous_pages_transcription)
        else:
            previous_section_transcription = "NA"
        for i, page_no in enumerate(tqdm(pages_list, initial=len(pages_completed), total=total_pages)):
            try:
                metadata = get_metadata(page_no, pdf_config.metadata_rules)
                record = {
                    "page_no": page_no,
                    "metadata": metadata
                }
                page_image_data_url = pdf_page_to_image(pdf_config.pdf_path, page_no, dpi=100)
                user_prompt = f"Previous section transcription:\n{previous_section_transcription}"
                llm_engine = self.llm_engine
                trancribed_page, response = llm_engine.generate(user_prompt, system_prompt=PDF_TRANSCRIBER_SYSTEM_PROMPT, image_urls=page_image_data_url, return_full_response=True)
                if trancribed_page is None:
                    logger.info(f"Main LLM couldn't transcribe page {page_no}! Using fallback model.")
                    llm_engine = self.fallback_llm_engine
                    trancribed_page, response = llm_engine.generate(user_prompt, system_prompt=PDF_TRANSCRIBER_SYSTEM_PROMPT, image_urls=page_image_data_url, return_full_response=True)
                    if trancribed_page is None:
                        trancribed_page = "<TRANSCRIPTION FAILED!>"
                        logger.error(f"Page {page_no} transcription failed! (LLM output is None)")
                if i + 1 < len(pages_list) and pages_list[i+1] - page_no == 1:
                    if '<CONT.>' in trancribed_page and previous_section_transcription != "NA":
                        combined_content = previous_section_transcription + '\n\n' + trancribed_page
                        previous_section_transcription = self._extract_partial_content(combined_content)
                    else:
                        previous_section_transcription = self._extract_partial_content(trancribed_page)
                else:
                    previous_section_transcription = ""
                previous_section_transcription = previous_section_transcription if previous_section_transcription != "" else "NA"
                record['transcription'] = trancribed_page
                record['llm_model'] = llm_engine.model_name
                record['total_tokens'] = response.usage.total_tokens
                write_jsonl(record, output_path=pdf_config.jsonl_path)
            except Exception:
                logger.exception(f"Page {page_no} transcription failed! Please resume the process manually!")
                return
        logger.info(
            f"Ingestion completed | pdf={pdf_config.pdf_path}"
        )