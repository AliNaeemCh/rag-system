FROM python:3.12-slim

WORKDIR /app

# install python dependencies
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# copy project
COPY app/ app/
COPY streamlit_app/ streamlit_app/
COPY run.py .
COPY start.sh .

EXPOSE 8501

# start script
CMD ["bash", "start.sh"]