import uvicorn

uvicorn.run(
    "reranker_service.app.main:app",
    host="0.0.0.0",
    port=8000,
)