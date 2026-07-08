from app.models import RetrievedDocument

import logging
import numpy as np

logger = logging.getLogger("rag.reranker")
logger.info("Loading file...")


class Reranker:
    def __init__(self, session, tokenizer, top_k: int):
        self.top_k = top_k
        self.session = session
        self.tokenizer = tokenizer

        # Get ONNX model input names
        self.input_names = {
            inp.name
            for inp in self.session.get_inputs()
        }

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        1. Create query-document pairs
        2. Tokenize pairs
        3. Run ONNX inference
        4. Attach scores
        5. Sort by relevance
        6. Return top-k docs
        """

        logger.info("Reranking started")

        if not docs:
            logger.warning("No documents received for reranking")
            return []

        # 1. Create query-document pairs
        pairs = [
            (query, doc.content)
            for doc in docs
        ]

        logger.debug(
            f"Prepared {len(pairs)} query-document pairs"
        )

        # 2. Tokenize pairs
        encoded = self.tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        # ONNX Runtime input preparation
        ort_inputs = {
            name: value.astype(np.int64)
            for name, value in encoded.items()
            if name in self.input_names
        }

        logger.debug(
            f"ONNX inputs: {list(ort_inputs.keys())}"
        )

        # 3. Run ONNX inference
        outputs = self.session.run(
            None,
            ort_inputs,
        )

        logits = outputs[0]

        # ms-marco-MiniLM-L12-v2 output shape:
        # (batch_size, 1)
        scores = logits.squeeze(-1).tolist()

        logger.debug(
            f"Reranker scores = {scores}"
        )

        # 4. Attach scores
        for i, score in enumerate(scores):
            docs[i].scores.reranker_score = score

        # 5. Sort by relevance
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        docs_sorted = [
            docs[i]
            for i in sorted_indices
        ]
        
        # 6. Return top-k
        reranked_docs = docs_sorted[:self.top_k]

        logger.info("Reranking completed")

        return reranked_docs