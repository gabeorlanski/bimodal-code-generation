FROM python:3.8-slim-buster
RUN apt-get update -y \
    && apt-get install -y git

RUN pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# Install dependencies:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    git clone https://github.com/gabeorlanski/taskio.git && \
    cd taskio && pip install -r requirements.txt && pip install -e .



ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY . .