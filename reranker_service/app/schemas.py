from pydantic import BaseModel


class Document(BaseModel):
    id: int
    content: str


class RerankRequest(BaseModel):
    query: str
    documents: list[Document]


class RankedScore(BaseModel):
    id: int
    score: float


class RerankResponse(BaseModel):
    scores: list[RankedScore]