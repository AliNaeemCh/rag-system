from app.ingestion.plain_text_to_chunks.config import ChunkingConfig
from app.ingestion.plain_text_to_chunks.utils.processors import join_texts
from app.ingestion.plain_text_to_chunks.utils.overlap import generate_overlap_texts
from app.ingestion.plain_text_to_chunks.utils.hierarchy import prepend_hierarchy
from app.core.tokenizer import Tokenizer
from app.core.utils import find_positions, replace_regex_pattern, combine_dicts, crossing_index
import re
from itertools import accumulate


class Chunker:
    """
    Core chunking engine:
    - merges sentences into token-limited chunks
    - handles hierarchy (<H1>, <H2>, etc.)
    - applies overlap between chunks
    """

    def __init__(self, tokenizer: Tokenizer, config: ChunkingConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def chunk_sections(self, sections_sentences: list[list[str]], metadata: list[dict] | None = None) -> list[str]:
        """
        CHUNKING METHODOLOGY:
        * Each chunk generally stays within the defined `chunk_size` token limit, with minor deviations occurring occasionally.
        * It is ensured that no chunk ends mid-sentence.
        * No two H1-level sections are merged into one chunk.
        * H2-level sections are also kept separate by default unless their size is too small in which case they are merged upto 50% of effective chunk size (`chunk_size-overlap_tokens_count`).
        * Sections below H2-level (H3, H4, etc.) are not separated and share the same chunks.
        * Chunk overlapping is also sentence-based.
        * Overlapping between chunks has tokens (`chunk_overlap_tokens`) based on the percentage of chunk size as defined by `chunk_size_pct`.
        * There is no overlap with the previous chunk when the previous chunk already fully contains a section and the current chunk begins a new section.
        * It is ensured that no chunk ends with only the title of a section, without its content.
        * Hierarchy of sections is prepended to the content in every chunk.
        * EDGE CASES:
            * If a sentence exceeds the effective chunk size (`chunk_size - overlap_tokens_count`), it is still fitted into the chunk,
                even if the resulting chunk becomes larger than the configured `chunk_size`.
            * If the last sentence of the previous chunk is longer than `chunk_overlap_tokens`,
                it is still carried over to the next chunk, even if this exceeds the overlap limit, as long as it fits within the chunk with room for at least one more sentence.
        """
        sections_sentences = sections_sentences.copy()
        overlap_tokens_count = round(self.config.chunk_size * (self.config.chunk_overlap_pct / 100))
        if self.config.separate_h2s:
            max_merging_threshold = round((self.config.chunk_size - overlap_tokens_count) * 0.5)    # 50% of effective chunk size
            # Makes sure too small sections (possibly titles only or one-liner intros) are are always merged with other sections
            abs_max_tokens_for_section_merging = 50    
        else:
            max_merging_threshold = float('inf')
            abs_max_tokens_for_section_merging = float('inf')
        all_sentences_counter = 0
        chunks = []
        # <For metadata handling>
        chunks_metadata = []
        section_sentences_count = [len(sentences) for sentences in sections_sentences]
        section_sentences_count_running_sums = list(accumulate(section_sentences_count))
        # <For metadata handling>
        j = 0
        len_sections_sentences = len(sections_sentences)
        next_section_total_tokens = None
        hierarchy = []
        total_long_sentences = 0
        while j < len_sections_sentences:
            if next_section_total_tokens is not None:
                current_section_total_tokens = next_section_total_tokens
            else:
                current_section_total_tokens = self._calc_total_section_tokens(sections_sentences[j])
            if j+1 < len_sections_sentences:
                next_section_total_tokens = self._calc_total_section_tokens(sections_sentences[j+1])
            # Allow sections to merge if total section size is below threshold and next section is not <H1>
            if j+1 < len_sections_sentences and \
            ((current_section_total_tokens + next_section_total_tokens) < max_merging_threshold
             or current_section_total_tokens <= abs_max_tokens_for_section_merging) \
                and "<H1>" not in sections_sentences[j+1][0]:
                sections_sentences[j+1] = sections_sentences[j] + sections_sentences[j+1]
                next_section_total_tokens = current_section_total_tokens + next_section_total_tokens
                j += 1
                continue
            if "<H1>" in sections_sentences[j][0]:
                hierarchy = []
            current_sentences = []
            current_tokens = 0
            i = 0
            while i < len(sections_sentences[j]):
                if not self.config.silent:
                    print(f'Now processing: Sentence {all_sentences_counter+1}')
                sentence = sections_sentences[j][i].strip(" ")
                if not sentence:
                    i += 1
                    all_sentences_counter += 1
                    continue
                h_tag_pos = find_positions(text=sentence, pattern=re.compile(r"<H\d+>"), find_first=True)
                if h_tag_pos is not None:
                    sentence = replace_regex_pattern(text=sentence, pattern=re.compile(r"<H(\d+)>"), replacement=r"H\1:")   # <Hx> -> Hx:
                sentence_token_count = self.tokenizer.count_tokens(sentence)
                # If sentence fits in current chunk
                # OR EDGE CASE: if the sentence is too long to fit in our chunk, we make exception and exceed the chunk size to fit this sentence
                if current_tokens + sentence_token_count <= self.config.chunk_size or sentence_token_count + overlap_tokens_count > self.config.chunk_size:
                    if sentence_token_count + overlap_tokens_count > self.config.chunk_size:
                        total_long_sentences += 1
                    if h_tag_pos is not None:
                        hierarchy.append(sentence.strip())
                    current_sentences.append(sentence)
                    current_tokens += sentence_token_count
                    i += 1
                    all_sentences_counter += 1

                else:
                    # Check if last sentence is of a subsection title
                    k = 1
                    while not current_sentences[-k].strip():
                        k += 1
                    if hierarchy[-1] == current_sentences[-k].strip():
                        # If last sentence is of a subsection title, remove it from current chunk and add it in the next one
                        current_sentences = current_sentences[:-k]
                        del hierarchy[-1]
                        all_sentences_counter -= k
                        i -= k
                    # Save current chunk
                    final_text = join_texts(current_sentences)
                    final_chunk = prepend_hierarchy(final_text, hierarchy)
                    chunks.append(final_chunk)
                    if metadata:
                        metadata_idx = crossing_index(section_sentences_count_running_sums, target=all_sentences_counter-1)
                        if metadata_idx > -1:
                            chunks_metadata.append(metadata[metadata_idx])
                        else:
                            chunks_metadata.append({})
                    else:
                        chunks_metadata.append({})
                    # Build overlap sentences for next chunk provided the last section/sub-section is NOT completed
                    k = 0
                    while not sections_sentences[j][i-k].strip():
                        k += 1
                    new_section_starts = find_positions(sections_sentences[j][i-k], pattern=re.compile(r"<H\d+>"), find_first=True) is not None
                    if not new_section_starts:
                        current_sentences_copy = current_sentences
                        current_sentences, current_tokens = generate_overlap_texts(current_sentences_copy, overlap_tokens_count, granularity=self.config.overlap_granularity, hierarchy=hierarchy)
                        if current_tokens == 0:
                            # EDGE CASE: If sentence-based granularity produces no overlapping, last sentence of the previous chunk will be added to the next one if it fits. Otherwise no overlapping
                            last_sentence_tokens = self.tokenizer.count_tokens(current_sentences_copy[-1])
                            next_sentence_tokens = self.tokenizer.count_tokens(sections_sentences[j][i])
                            if last_sentence_tokens + next_sentence_tokens <= self.config.chunk_size:
                                current_sentences = [current_sentences_copy[-1]]
                                current_tokens = last_sentence_tokens
                            else:
                                current_sentences = []
                                current_tokens = 0
                    else:
                        current_sentences = []
                        current_tokens = 0

            # Add final chunk
            if current_sentences:
                final_text = join_texts(current_sentences)
                final_chunk = prepend_hierarchy(final_text, hierarchy)
                chunks.append(final_chunk)
                if metadata:
                    metadata_idx = crossing_index(section_sentences_count_running_sums, target=all_sentences_counter-1)
                    if metadata_idx > -1:
                        chunks_metadata.append(metadata[metadata_idx])
                    else:
                        chunks_metadata.append({})
                else:
                    chunks_metadata.append({})
            j += 1
        if not self.config.silent:
            print(f'\nTotal long sentences: {total_long_sentences}')
        if metadata:
            return chunks, chunks_metadata
        return chunks

    def _calc_total_section_tokens(self, section_sentences: list[str]):
        total_section_tokens = 0
        for s in section_sentences:
            s = s.strip(" ")
            if not s:
                continue
            total_section_tokens += self.tokenizer.count_tokens(s)
        return total_section_tokens