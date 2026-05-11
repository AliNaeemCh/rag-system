from tqdm import tqdm
from app.ingestion.plain_text_to_chunks.utils.parser import iter_sections
from app.ingestion.plain_text_to_chunks.chunker import Chunker
from app.ingestion.plain_text_to_chunks.config import ChunkingConfig
from app.core.utils import write_jsonl
from app.core.tokenizer import tokenizer
from app.ingestion.plain_text_to_chunks.utils.nlp import nlp
from pathlib import Path

def run_pipeline(chunking_config: ChunkingConfig, input_path: Path, output_path: Path):

    pbar = tqdm(total=100)
    chunker = Chunker(tokenizer=tokenizer, config=chunking_config)

    sections_sentences = []

    for output in iter_sections(input_path):

        section = output["section"]
        metadata = output['metadata']
        progress = output["progress"]

        pbar.n = int(progress * 100)
        pbar.refresh()

        if "<H1>" in section and sections_sentences:
            chunks = chunker.chunk_sections(sections_sentences)
            data = [
                {
                    "content": chunk,
                    "metadata": metadata
                }
                for chunk in chunks
            ]
            write_jsonl(data, output_path)
            sections_sentences = []
        lines = section.split('\n')
        lines = [t + '\n' for t in lines[:-1]] + [lines[-1]]
        sentences = []
        for line in lines:
            doc = nlp(line)
            sentences.extend([sent.text for sent in doc.sents])
        sections_sentences.append(sentences)
    pbar.close()

chunking_config = ChunkingConfig(
    
)