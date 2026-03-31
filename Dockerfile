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
    && rm -rf /var/lib/apt/lists/*

# Build and install RNNoise (system-wide)
RUN git clone https://git.xiph.org/rnnoise.git /tmp/rnnoise \
    && cd /tmp/rnnoise \
    && ./autogen.sh \
    && ./configure \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/rnnoise

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
