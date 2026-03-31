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
        libopenblas-dev \
        gfortran \
        libtool-bin \
        m4 \
    && rm -rf /var/lib/apt/lists/*

# RNNoise removed: no native build step.

# Copy project into image
# Copy project into image
COPY . /app

# Ensure start script is executable (copied from repo root)
RUN if [ -f /app/start.sh ]; then chmod +x /app/start.sh; fi

# Install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ -f /app/requirements.docker.txt ]; then pip install --no-cache-dir -r /app/requirements.docker.txt; fi

# Default env vars — override in Railway or your env
ENV PORT=8080

EXPOSE 8080

# Run the app (Railway provides $PORT at runtime)
CMD ["/app/start.sh"]
