# RSABias on current Python, with the exact dependency versions of uv.lock.
#
# Build:  docker build -t rsabias --build-arg VCS_REF=$(git rev-parse --short HEAD) .
# Run:    docker run --rm -v "$PWD:/work" rsabias -a classify -i key.pem -o .
#         (/work is the working directory; add --user "$(id -u):$(id -g)" if your UID is not 1000)
#
# The legacy environment of the paper is built from the tag esorics2020-env, see README.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

RUN pip install --no-cache-dir uv==0.12.22
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /src
# Dependencies first, so code changes do not reinstall them
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY rsabias ./rsabias
# Non-editable: the package and its model data are copied into the environment
RUN uv sync --frozen --no-dev --no-editable


FROM ${PYTHON_IMAGE}

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="rsabias" \
      org.opencontainers.image.description="Origin attribution of RSA private keys (ESORICS 2020), on current Python and dependencies (uv.lock)." \
      org.opencontainers.image.source="https://github.com/crocs-muni/RSABias" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT" \
      rsabias.environment="current"

COPY --from=builder /opt/venv /opt/venv
RUN useradd --uid 1000 --create-home rsabias

# MPLCONFIGDIR: matplotlib needs a writable cache directory, also for UIDs without a home
ENV PATH=/opt/venv/bin:$PATH \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

USER rsabias
WORKDIR /work
ENTRYPOINT ["rsabias"]
CMD ["--help"]
