# Backend-only image: FastAPI voice server.
# Stage 1 keeps the pattern in case we add a Rust/Go build later.

FROM python:3.11-slim

# ponytail: capture the build commit so /version can prove repo == deployed.
# Railway auto-injects $GIT_COMMIT_SHA via Nixpacks, but nixpacks.toml isn't
# authoritative here (we use Dockerfile builds) — pass it explicitly via
#   docker build --build-arg GIT_SHA=$(git rev-parse HEAD) ...
# or set GIT_SHA in Railway's service Variables. Falls back to "unknown" so a
# local dev container still boots instead of crashing on an unset env.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential autoconf automake libtool pkg-config git ca-certificates \
        libsndfile1-dev ffmpeg curl wget libopenblas-dev gfortran libtool-bin m4 \
    && rm -rf /var/lib/apt/lists/*

# ponytail: dedicated non-root user. The image's default user is root,
# which amplifies the blast radius of any RCE in Python deps or the
# runtime. A dedicated unprivileged user + writable /app only costs a
# few lines and aligns with the scanner's "container runs as default
# root user" finding.
RUN groupadd --system --gid 1001 app \
    && useradd  --system --uid 1001 --gid app --home /app --shell /usr/sbin/nologin app

COPY --chown=app:app . /app/

RUN if [ -f /app/start.sh ]; then chmod +x /app/start.sh; fi

RUN python -m pip install --upgrade 'pip>=22' 'setuptools>=70,<81' wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && if [ -f /app/requirements.docker.txt ]; then pip install --no-cache-dir -r /app/requirements.docker.txt; fi

# Drop privileges for the runtime. CMD runs as `app` (uid 1001).
USER app

ENV PORT=8080
EXPOSE 8080

CMD ["/app/start.sh"]
