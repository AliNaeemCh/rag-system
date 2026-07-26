FROM python:3.12-slim

WORKDIR /root

COPY reranker_service/requirements.txt reranker_service/requirements.txt
RUN pip install --no-cache-dir -r reranker_service/requirements.txt

COPY reranker_service/ reranker_service/
COPY models/cross_encoder/ms-marco-MiniLM-L12-v2_onnx models/cross_encoder/ms-marco-MiniLM-L12-v2_onnx

EXPOSE 8000

ENV PYTHONPATH=/root

CMD ["python", "reranker_service/run.py"]