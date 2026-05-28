from app.ingestion.chunks_generation.config import ChunkingConfig
from app.ingestion.chunks_generation.utils.processors import join_texts
from app.ingestion.chunks_generation.utils.overlap import generate_overlap_texts
from app.ingestion.chunks_generation.utils.hierarchy import prepend_hierarchy
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
        * Each chunk always stays within the defined `chunk_size` token limit.
        * It is ensured that no chunk ends mid-sentence.
        * No two H1-level sections are merged into one chunk.
        * H2-level sections are also kept separate by default unless their size is too small in which case they are merged upto 50% of effective chunk size (`chunk_size-overlap_tokens_count`).
        * Sections below H2-level (H3, H4, etc.) are not separated and share the same chunks.
        * Chunk overlapping is also sentence-based.
        * Overlapping between chunks has tokens (`chunk_overlap_tokens`) based on the percentage of chunk size as defined by `chunk_size_pct`.
        * By default, there is no cross-section overlap (The overlap text can't include a section other than the current.)
        * By default, there is no overlap with the previous chunk when the previous chunk already fully contains a section and the current chunk begins a new section.
        * It is ensured that no chunk ends with only the title of a section, without its content.
        * Hierarchy of sections is prepended to the content in every chunk.
        * EDGE CASES:
            * If a sentence exceeds the effective chunk size (`chunk_size - overlap_tokens_count`), it is still fitted into the chunk,
                even if the resulting chunk becomes larger than the configured `chunk_size`.
            * If the last sentence of the previous chunk is longer than `chunk_overlap_tokens`,
                it is still carried over to the next chunk, even if this exceeds the overlap limit, as long as it fits within the chunk with room for at least one more sentence.
        """
        counter = 0
        def finalize_chunk():
            final_text = join_texts(current_sentences)
            final_chunk = prepend_hierarchy(final_text, hierarchy)
            # print('FINALIZED CHUNK IS: ', final_chunk)
            chunks.append(final_chunk)
            # Metadata
            final_chunk_tokens = self.tokenizer.count_tokens(final_chunk)
            additional_metadata = {
                'total_tokens': final_chunk_tokens,
                'total_overlap_tokens': total_overlap_tokens
            }
            if metadata:
                metadata_idx = crossing_index(section_sentences_count_running_sums, target=all_sentences_counter-1)
                if metadata_idx > -1:
                    chunks_metadata.append(metadata[metadata_idx] | additional_metadata)
                else:
                    chunks_metadata.append(additional_metadata)
            else:
                chunks_metadata.append(additional_metadata)

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
            total_overlap_tokens = 0
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
                hierarchy_for_extended_text = hierarchy
                if h_tag_pos is not None:
                    sentence = replace_regex_pattern(text=sentence, pattern=re.compile(r"<H(\d+)>"), replacement=r"H\1:")   # <Hx> -> Hx:
                    sentence_h_tag_level = int(sentence.split(":")[0][1:])
                    h_idx = len(hierarchy) - 1
                    if h_idx >= 0:
                        while h_idx >= 0:
                            h_level = int(hierarchy[h_idx].split(":")[0][1:])
                            if h_level != sentence_h_tag_level:
                                break
                            h_idx -= 1
                        hierarchy_for_extended_text = hierarchy[:h_idx + 1]
                        if h_idx < len(hierarchy) - 1:
                            # print('HIERARCHY MUTATED')
                            # print('CURRENT SENTENCE: ', sentence)
                            # print('ORIGINAL HIERARCHY', hierarchy)
                            # print('MOD HIERARCHY', hierarchy_for_extended_text)
                            pass
                sentence_token_count = self.tokenizer.count_tokens(sentence)
                extended_text = prepend_hierarchy(text=join_texts(current_sentences + [sentence]), hierarchy=hierarchy_for_extended_text)
                extended_text_token_count = self.tokenizer.count_tokens(extended_text)
                # if counter > 100:
                #     return
                # print('CURRENT SENTENCES IS: ', current_sentences)
                # print('extended_text IS: ', extended_text)
                # print("TOKEN COUNT:", extended_text_token_count)
                # If sentence fits in current chunk
                # OR EDGE CASE: if the sentence is too long to fit in our chunk, we make exception and exceed the chunk size to fit this sentence
                if extended_text_token_count <= self.config.chunk_size or sentence_token_count + overlap_tokens_count > self.config.chunk_size:
                    # print('ENTERED APPENDING')
                    if sentence_token_count + overlap_tokens_count > self.config.chunk_size:
                        total_long_sentences += 1
                    if h_tag_pos is not None:
                        hierarchy.append(sentence.strip())
                    current_sentences.append(sentence)
                    i += 1
                    all_sentences_counter += 1

                else:
                    # print('ENTERED FINALIZE CHUNK')
                    # Check if last sentence is of a subsection title
                    k = 1
                    while not current_sentences[-k].strip():
                        k += 1
                    if hierarchy and hierarchy[-1] == current_sentences[-k].strip():
                        # If last sentence is of a subsection title, remove it from current chunk and add it in the next one
                        current_sentences = current_sentences[:-k]
                        del hierarchy[-1]
                        all_sentences_counter -= k
                        i -= k
                    # Save current chunk with metadata
                    finalize_chunk()

                    # Build overlap sentences for next chunk
                    if overlap_tokens_count > 0:
                        k = 0
                        while not sections_sentences[j][i-k].strip():
                            k += 1
                        new_section_starts = find_positions(sections_sentences[j][i-k], pattern=re.compile(r"<H\d+>"), find_first=True) is not None
                        if not new_section_starts or self.config.cross_section_overlap:
                            current_sentences_copy = current_sentences
                            current_sentences, _ = generate_overlap_texts(current_sentences_copy, overlap_tokens_count, granularity=self.config.overlap_granularity,
                                                                                       cross_section_overlap=self.config.cross_section_overlap, hierarchy=hierarchy)
                            stripped_overlap_tokens = sum([self.tokenizer.count_tokens(s.strip()) for s in current_sentences])
                            if stripped_overlap_tokens == 0:
                                # EDGE CASE: If no overlapping is produced, last sentence of the previous chunk will be added to the next one if it fits. Otherwise no overlapping
                                k = 1
                                last_sentence_tokens = 0
                                # Makes sure last sentence is picked even if the last element of `current_sentences` is a new line (\n) or a blank space
                                while True:
                                    last_sentence_tokens += self.tokenizer.count_tokens(current_sentences_copy[-k])
                                    if current_sentences_copy[-k].strip():
                                        break
                                    k += 1
                                t = 0
                                next_sentence_tokens = 0
                                # Makes sure proper next sentence is picked and not a new line (\n) or a blank space
                                while i + t < len(sections_sentences[j]):
                                    next_sentence_tokens += self.tokenizer.count_tokens(sections_sentences[j][i+t])
                                    if sections_sentences[j][i+t].strip():
                                        break
                                    t += 1
                                if last_sentence_tokens + next_sentence_tokens <= self.config.chunk_size:
                                    current_sentences = current_sentences_copy[-k:]
                                else:
                                    current_sentences = []
                        else:
                            current_sentences = []
                        total_overlap_tokens = self.tokenizer.count_tokens(join_texts(current_sentences))
                    else:
                        current_sentences, total_overlap_tokens = [], 0
                counter += 1
            # Add final chunk
            if current_sentences:
                finalize_chunk()
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