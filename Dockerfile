FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    git \
    libsndfile1-dev \
    libsndfile1 \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN python3 -m pip install --no-cache-dir \
    accelerate==1.10.1 \
    datasets==3.3.2 \
    evaluate==0.4.5 \
    huggingface-hub==0.34.4 \
    librosa==0.11.0 \
    transformers==4.50.0 \
    wandb==0.24.0 \
    matplotlib==3.10.0 \
    scikit-learn==1.7.1 \
    hf_transfer

RUN python3 -m pip install --no-cache-dir \
    numpy==2.1.2 \
    pandas==2.3.2 \
    scipy==1.15.3 \
    soundfile==0.13.1 \
    tqdm==4.67.1 \
    pillow==11.0.0


ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY train_model.py /app/train_model.py

CMD ["python3", "train_model.py"]
