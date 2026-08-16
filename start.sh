#!/bin/bash

echo "Starting FastAPI backend..."
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
wait