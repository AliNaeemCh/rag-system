import numpy as np


class Reranker:

    def __init__(self, session, tokenizer):

        self.session = session
        self.tokenizer = tokenizer

        self.input_names = {
            inp.name
            for inp in session.get_inputs()
        }

    def rerank(
        self,
        query,
        docs
    ):

        pairs = [
            (query, d.content)
            for d in docs
        ]

        encoded = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        ort_inputs = {
            k: v.astype(np.int64)
            for k, v in encoded.items()
            if k in self.input_names
        }

        logits = self.session.run(
            None,
            ort_inputs,
        )[0]

        scores = logits.squeeze(-1).tolist()

        return [
            {
                "id": doc.id,
                "score": score,
            }
            for doc, score in zip(docs, scores)
        ]