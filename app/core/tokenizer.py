import tiktoken

class Tokenizer:
    def __init__(self, encoding_name: str = "o200k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str):
        return self.encoding.encode(text)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


tokenizer = Tokenizer()