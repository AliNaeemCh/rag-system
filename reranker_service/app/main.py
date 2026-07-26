from fastapi import FastAPI
from reranker_service.app.api import router

app = FastAPI()
app.include_router(router)