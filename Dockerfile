# Stage 1: build the React frontend
FROM node:20-alpine AS frontend

WORKDIR /fe
COPY AgentsAi_Frontend/package.json AgentsAi_Frontend/package-lock.json ./
RUN npm ci
COPY AgentsAi_Frontend/ ./
RUN npm run build

# Stage 2: Python backend with the built frontend baked in
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential autoconf automake libtool pkg-config git ca-certificates \
        libsndfile1-dev ffmpeg curl wget libopenblas-dev gfortran libtool-bin m4 \
    && rm -rf /var/lib/apt/lists/*

# Backend code
COPY Agent_IA_Server/ /app/

RUN if [ -f /app/start.sh ]; then chmod +x /app/start.sh; fi

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && if [ -f /app/requirements.docker.txt ]; then pip install --no-cache-dir -r /app/requirements.docker.txt; fi

# Build artifacts from the FE stage, served by FastAPI at "/"
COPY --from=frontend /fe/dist /app/static/fe

ENV PORT=8080
EXPOSE 8080

CMD ["/app/start.sh"]
