# Dockerfile: builds RNNoise C library and application image
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install build and runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        git \
        ca-certificates \
        libsndfile1-dev \
        ffmpeg \
        curl \
        wget \
        libtool-bin \
        m4 \
    && rm -rf /var/lib/apt/lists/*

# Build and install RNNoise (system-wide)
# Use GitHub mirror for reliability (git.xiph.org often fails DNS)
# Build and install RNNoise (system-wide) with defensive steps
RUN set -eux; \
    git clone --depth 1 https://github.com/xiph/rnnoise.git /tmp/rnnoise || (echo "git clone failed, trying tarball" && curl -L https://github.com/xiph/rnnoise/archive/refs/heads/master.tar.gz | tar xz -C /tmp && mv /tmp/rnnoise-master /tmp/rnnoise); \
    cd /tmp/rnnoise; \
    ./autogen.sh || true; \
    if [ -f configure ]; then ./configure; else echo "configure not found, attempting autoreconf -i" && autoreconf -i && ./configure; fi; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    rm -rf /tmp/rnnoise

# Copy project into image
COPY . /app

# Install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Default env vars — override in Railway or your env
ENV RNNOISE_ENABLED=true \
    RNNOISE_FALLBACK_ENABLED=true \
    PORT=8000

EXPOSE 8000

# Run the app (Railway provides $PORT at runtime)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
