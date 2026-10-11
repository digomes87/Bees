# Runs the bee-migration command, or serves the dashboard (see docker-compose.yml).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# The dashboard loads the package as a wheel next to its pages.
COPY docs ./docs
RUN pip wheel . --no-deps --wheel-dir docs/package

RUN useradd --create-home bee && mkdir /work && chown bee:bee /work
USER bee
WORKDIR /work

ENTRYPOINT ["bee-migration"]
CMD ["--help"]
