FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/worker

RUN set -eux; \
    for i in 1 2 3 4 5; do \
      apt-get update && break || (echo "apt-get update failed, retry $i" && sleep 5); \
    done; \
    for i in 1 2 3 4 5; do \
      apt-get install -y --no-install-recommends --fix-missing \
        python3 python3-pip python3-venv git ca-certificates curl && break \
      || (echo "apt-get install failed, retry $i" && sleep 8); \
    done; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /worker
COPY requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN python3 -c "import torch; print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available()); print('arch_list', torch.cuda.get_arch_list() if torch.cuda.is_available() else None)"

COPY . .
CMD ["python3", "ml_worker.py"]
