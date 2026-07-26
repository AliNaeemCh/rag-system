#!/bin/bash

echo "Starting FastAPI backend..."
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 &

echo "Starting Streamlit frontend..."
streamlit run streamlit_app/main.py \
    --server.address=0.0.0.0 \
    --server.port=8501

wait