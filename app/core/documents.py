class Document:
    def __init__(self, text: str, metadata: dict | None = None):
        self.text = text
        self.metadata = metadata or {}