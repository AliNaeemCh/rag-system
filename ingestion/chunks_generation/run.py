from app.core.logger import setup_logging
setup_logging()

from ingestion.chunks_generation.utils.parsers import iter_sections
from ingestion.chunks_generation.chunker import Chunker
from analysis.chunks.generate_chunks_stats import generate_chunks_stats
from ingestion.chunks_generation.config import ChunkingConfig
from app.core.utils import write_jsonl, reset_jsonl, find_positions, extract_last_jsonl_object, load_jsonl, build_jsonl_index
from app.core.tokenizer import tokenizer
from ingestion.chunks_generation.utils.nlp import nlp
from app.core.config import settings

import logging
logger = logging.getLogger("ingestion.chunks_generation.run")
logger.info("Loading file...")

from tqdm import tqdm
from pathlib import Path
import logging
import re

def run_pipeline(chunking_config: ChunkingConfig, input_path: Path, output_path: Path):
    try:
        pbar = tqdm(total=100)
        chunker = Chunker(tokenizer=tokenizer, config=chunking_config)
        sections_sentences = []
        metadata = []
        h_tags = ["<H1>", "<H2>"] if chunking_config.separate_h2s else ["<H1>"]
        last_chunk = extract_last_jsonl_object(jsonl_path=output_path)
        chunks_counter = last_chunk.get('chunk_id', 0)

        if chunking_config.resume:
            if chunks_counter > 0:
                logger.info("Chunking Resumed")
            else:
                logger.info("Chunking Started")
        else:
            while True:
                user_in = input(f"\033[93mWarning:\033[0m Previously processed documents ({chunks_counter}) will be deleted. Type 'confirm' to proceed: ")
                if user_in == "confirm":
                    reset_jsonl(output_path)
                    break
                else:
                    print("Invalid input. Try again!")
            reset_jsonl(jsonl_path=output_path)
            last_chunk = None
            logger.info("Chunking Started")

        def chunk_and_save():
            nonlocal chunks_counter
            chunks, chunks_metadata = chunker.chunk_sections(sections_sentences, metadata=metadata)
            data = []
            for i, chunk in enumerate(chunks):
                chunks_counter += 1
                data.append(
                    {
                        "chunk_id": chunks_counter,
                        "metadata": chunks_metadata[i],
                        "content": chunk
                    }
                )
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
        logger.exception("Chunking Failed! Please resume the process manually.")
        return

chunking_config = ChunkingConfig(resume=False)  # Default params

chunks_file_name = "sys_annual_2025_chunks.jsonl"

# run_pipeline(
#     chunking_config=chunking_config,
#     input_path=settings.PROCESSED_DATA_DIR / "SYS Limited Annual - 2025.jsonl",
#     output_path=settings.PROCESSED_DATA_DIR / chunks_file_name
# )

# Chunks jsonl file index generation

chunks_jsonl_file_index_path = settings.PROCESSED_DATA_DIR / "chunks_jsonl_index.pkl"
build_jsonl_index(jsonl_path=settings.PROCESSED_DATA_DIR / chunks_file_name, index_path=chunks_jsonl_file_index_path)
print(f"\nChunks jsonl file index saved to: {chunks_jsonl_file_index_path}")

# Chunks stats generation

generate_chunks_stats(
    path=settings.PROCESSED_DATA_DIR / chunks_file_name,
    chunk_size=chunking_config.chunk_size,
    chunk_overlap_pct=chunking_config.chunk_overlap_pct
)