# Backend-only image: FastAPI voice server.
# Stage 1 keeps the pattern in case we add a Rust/Go build later.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential autoconf automake libtool pkg-config git ca-certificates \
        libsndfile1-dev ffmpeg curl wget libopenblas-dev gfortran libtool-bin m4 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN if [ -f /app/start.sh ]; then chmod +x /app/start.sh; fi

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && if [ -f /app/requirements.docker.txt ]; then pip install --no-cache-dir -r /app/requirements.docker.txt; fi

ENV PORT=8080
EXPOSE 8080

CMD ["/app/start.sh"]
