FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime

LABEL version="0.7"

USER root

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    vim \
    git-all \
    python3-dev \
    python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    scipy \
    tensorboard \
    scikit-learn \
    umap-learn \
    pandas \
    onnx \
    onnxscript \
    tqdm \
    pyyaml \
    matplotlib

RUN python -c "import torch, umap; print(torch.__version__); print(torch.cuda.is_available()); print(umap.__version__)"
