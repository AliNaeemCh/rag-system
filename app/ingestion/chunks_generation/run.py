from tqdm import tqdm
from app.ingestion.chunks_generation.utils.parsers import iter_sections, extract_last_chunk
from app.ingestion.chunks_generation.chunker import Chunker
from app.analysis.chunks.generate_chunks_stats import generate_chunks_stats
from app.ingestion.chunks_generation.config import ChunkingConfig
from app.core.utils import write_jsonl, reset_jsonl, find_positions
from app.core.tokenizer import tokenizer
from app.ingestion.chunks_generation.utils.nlp import nlp
from pathlib import Path
from app.core.config import settings
import logging
from app.core.logger import setup_logging
import re
setup_logging()
logger = logging.getLogger("app.ingestion.chunks_generation.run")

def run_pipeline(chunking_config: ChunkingConfig, input_path: Path, output_path: Path):
    try:
        pbar = tqdm(total=100)
        chunker = Chunker(tokenizer=tokenizer, config=chunking_config)
        sections_sentences = []
        metadata = []
        h_tags = ["<H1>", "<H2>"] if chunking_config.separate_h2s else ["<H1>"]
        if chunking_config.resume:
            last_chunk = extract_last_chunk(jsonl_path=output_path)
            if last_chunk:
                logger.info("Chunking Resumed")
            else:
                logger.info("Chunking Started")
        else:
            reset_jsonl(jsonl_path=output_path)
            last_chunk = None
            logger.info("Chunking Started")

        def chunk_and_save():
            chunks, chunks_metadata = chunker.chunk_sections(sections_sentences, metadata=metadata)

            data = [
                {
                    "metadata": chunks_metadata[i],
                    "content": chunk
                }
                for i, chunk in enumerate(chunks)
            ]
            write_jsonl(data, output_path)

        for output in iter_sections(input_path, h_tags=h_tags, last_chunk=last_chunk):
            section = output["section"]
            progress = output["progress"]
            pbar.n = int(progress * 100)

            if "<H1>" in section and sections_sentences:
                chunk_and_save()
                sections_sentences = []
                metadata = []
            # Gathering all sub-sections of an H1-level section
            lines = section.split('\n')
            lines = [t + '\n' for t in lines[:-1]] + [lines[-1]]
            sentences = []
            for line in lines:
                h_tag_pos = find_positions(line, pattern=re.compile(r"<H\d+>"), find_first=True)
                if h_tag_pos is not None:
                    # Makes sure the heading names are not chopped by sentence tokenizer
                    sentences.extend([line])
                else:
                    doc = nlp(line)
                    sentences.extend([sent.text for sent in doc.sents])
            sections_sentences.append(sentences)
            metadata.append(output['metadata'].copy())
            pbar.refresh()

        if sections_sentences:
            chunk_and_save()
        pbar.close()
        logger.info("Chunking Completed")

    except Exception:
        logger.exception("Chunking Failed!")

chunking_config = ChunkingConfig(resume=False)  # Default params

chunks_file_name = "sys_annual_2025_chunks.jsonl"

run_pipeline(
    chunking_config=chunking_config,
    input_path=settings.PROCESSED_DATA_DIR / "SYS Limited Annual - 2025.jsonl",
    output_path=settings.PROCESSED_DATA_DIR / chunks_file_name
)

generate_chunks_stats(
    path=settings.PROCESSED_DATA_DIR / chunks_file_name,
    chunk_size=chunking_config.chunk_size,
    chunk_overlap_pct=chunking_config.chunk_overlap_pct
)