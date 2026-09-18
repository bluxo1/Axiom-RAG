# Axiom backend container (Phases.md Phase 0).
#
# Python 3.11 is the pinned floor from Prompt.md §TECH STACK. The image carries
# no source tree: the package is installed, so `app` resolves the same way in
# the container as it does in CI.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/axiom

# Dependency metadata first so a source-only edit reuses the install layer.
COPY pyproject.toml ./
COPY app ./app
RUN pip install .

# `config.yaml` is mounted read-only at /srv/axiom/config.yaml by compose; the
# path is passed explicitly so nothing depends on the working directory.
ENV CONFIG_PATH=/srv/axiom/config.yaml

# Ingested corpora land here (bind-mounted in dev).
RUN mkdir -p /srv/axiom/data \
    && useradd --create-home --uid 10001 axiom \
    && chown -R axiom:axiom /srv/axiom
USER axiom

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
