# Default to CPU base
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# ARG to select requirements file
ARG REQ_FILE=requirements.txt
COPY ${REQ_FILE} /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . .

EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", "-w", "2", "-b", "0.0.0.0:8000"]
