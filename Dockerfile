# Reference environment for RSABias with the original (ESORICS 2020) dependencies.
#
# Build:  docker build -t rsabias:esorics2020 --build-arg VCS_REF=$(git rev-parse --short HEAD) .
# Run:    docker run --rm -v "$PWD/out:/out" rsabias:esorics2020 \
#             -a classify -i rsabias/model/sample_private_key.pem -o /out
# Data:   mount datasets as volumes, e.g. -v /path/to/dataset:/data:ro

# pandas 0.24.2 only has wheels up to Python 3.7 and does not compile on 3.8,
# which pins the interpreter. See docker/requirements-esorics2020.txt.
ARG PYTHON_IMAGE=python:3.7-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

# gmpy2 2.0.8 has no wheels and is compiled against GMP/MPFR/MPC.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgmp-dev libmpfr-dev libmpc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-esorics2020.txt /tmp/requirements.txt
RUN pip wheel --no-cache-dir -w /wheels -r /tmp/requirements.txt


FROM ${PYTHON_IMAGE}

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="rsabias (legacy ESORICS 2020 environment)" \
      org.opencontainers.image.description="Legacy reference environment: Python 3.7 with the original 2020 dependency pins (docker/requirements-esorics2020.txt). Unmaintained, EOL stack; intended for reproducing the paper's results, not for production use." \
      org.opencontainers.image.version="esorics2020" \
      org.opencontainers.image.source="https://github.com/crocs-muni/RSABias" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT" \
      rsabias.environment="legacy"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgmp10 libmpfr6 libmpc3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY docker/requirements-esorics2020.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements.txt \
    && rm -rf /wheels

WORKDIR /opt/rsabias
COPY . .
# Run from the source tree rather than installing: setup.py pins the bumped
# numpy/scipy, which the generated console script would insist on, and
# MANIFEST.in does not package every model file (e.g. distributions/).
ENV PYTHONPATH=/opt/rsabias \
    MPLBACKEND=Agg
ENTRYPOINT ["python", "-m", "rsabias.cli"]
CMD ["--help"]
