from dataclasses import dataclass

@dataclass(slots=True)
class RetrievedDocument:
    id: int
    content: str
    metadata: dict
    distance: float