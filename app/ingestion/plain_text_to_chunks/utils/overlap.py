from app.ingestion.plain_text_to_chunks.config import OverlapGranularity
from app.ingestion.plain_text_to_chunks.utils.processors import join_texts
from app.core.tokenizer import tokenizer


def generate_overlap_texts(
    sentences: list[str],
    overlap_tokens_count: int,
    granularity: OverlapGranularity,
    hierarchy: list[str] | None = None
) -> tuple[list[str], int]:
    overlap_texts = []
    tokens_used = 0
    end = False
    # Start from end of current chunk and work backwards
    for sentence in reversed(sentences):
        if hierarchy is not None and sentence.strip() in hierarchy:
            break
        sentence_tokens_count = tokenizer.count_tokens(sentence)
        if tokens_used + sentence_tokens_count <= overlap_tokens_count:
            overlap_texts.insert(0, sentence)  # insert full sentence
            tokens_used += sentence_tokens_count
        elif granularity == OverlapGranularity.WORD_BASED:
            words = sentence.split()
            sub_sentence = ''
            for i in range(len(words) - 1, -1, -1):
                last_words = join_texts(words[i:])
                last_words_tokens_count = tokenizer.count_tokens(last_words)
                if tokens_used + last_words_tokens_count <= overlap_tokens_count:
                    sub_sentence = last_words
                else:
                    overlap_texts.insert(0, sub_sentence)
                    tokens_used += last_words_tokens_count
                    end = True
                    break
            if end:
                break
        else:
            break
    return overlap_texts, tokens_used