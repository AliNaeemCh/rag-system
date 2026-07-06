FROM python:3.12-slim

WORKDIR /app

# system dependencies (important for ML)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY app/ app/
COPY streamlit_app/ streamlit_app/
COPY run.py .
COPY models/cross_encoder/ models/cross_encoder/
COPY start.sh .

EXPOSE 8000
EXPOSE 8501

# start script
CMD ["bash", "start.sh"]