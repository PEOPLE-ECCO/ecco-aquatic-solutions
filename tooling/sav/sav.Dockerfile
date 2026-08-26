ARG BASE_IMAGE=ghcr.io/people-ecco/ecco-algorithm-base:1.1.1

FROM alpine:3.20 AS fetcher
RUN apk add --no-cache curl
WORKDIR /out
COPY tooling/sav/model_downloader.sh .
RUN sh model_downloader.sh && rm model_downloader.sh

FROM ${BASE_IMAGE}

# Algorithm base path
ARG ALGORITHM_BASE="sav.main"
ENV ALGORITHM_BASE=${ALGORITHM_BASE}
ENV UV_PYTHON=3.12

# metadata
ARG SOLUTION="PEOPLE-ECCO Submerged Aquatic Vegetation"
ENV SOLUTION=${SOLUTION}

ARG SOLUTION_VERSION="1.0.1"
ENV SOLUTION_VERSION=${SOLUTION_VERSION}

LABEL org.opencontainers.image.title=${SOLUTION} \
      org.opencontainers.image.description=${SOLUTION} \
      org.opencontainers.image.version=${SOLUTION_VERSION} \
      org.opencontainers.image.authors="Gyde Krueger <gykr@dhigroup.com>, Paul Jacques Connetable <paco@dhigroup.com>" \
      org.opencontainers.image.vendor="DHI" \
      org.opencontainers.image.source="https://github.com/PEOPLE-ECCO/ecco-marine-solutions" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY tooling/sav/requirements.txt requirements.txt
RUN uv pip install --system -r requirements.txt

# Copy sourcecode. Copy models separately to allow for Docker caching
COPY --exclude=src/models/*.pkl src/ sav/
COPY --from=fetcher out/*.pkl sav/models/
