# syntax=docker/dockerfile:1
# BackTalk backend node. Build from the REPO ROOT (it needs train/ + models):
#   docker build -f backend-node/Dockerfile -t backtalk-node .
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir fastapi "uvicorn[standard]" numpy tqdm tokenizers psutil

# Model definition + the worker.
COPY train/ ./train/
COPY backend-node/ ./backend-node/

# Bake the served checkpoints + tokenizer so the node runs with no MODEL_SOURCE.
# (Omit these and set MODEL_SOURCE to fetch them at first start instead.)
COPY checkpoints/backtalk-assistant/ ./checkpoints/backtalk-assistant/
COPY checkpoints/backtalk-assistant-v2/ ./checkpoints/backtalk-assistant-v2/
COPY checkpoints/backtalk-storyteller/ ./checkpoints/backtalk-storyteller/
COPY tokenizers/backtalk-tokenizer/ ./tokenizers/backtalk-tokenizer/

CMD ["python3", "backend-node/worker.py"]
