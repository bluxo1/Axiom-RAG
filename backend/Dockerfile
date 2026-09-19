# Axiom backend container (Phases.md Phase 0, deploy in Phase 4).
#
# Python 3.11 is the pinned floor from Prompt.md §TECH STACK. The image carries
# no source tree: the package is installed, so `app` resolves the same way in
# the container as it does in CI.
#
# Build context is the *repository root* (see docker-compose.yml and render.yaml)
# so the committed `config.yaml` can be baked in — the image boots standalone on
# a PaaS (Render) with no bind mount. In dev, compose still mounts config.yaml
# read-only over the baked copy so knob edits take effect without a rebuild.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/axiom

# Dependency metadata first so a source-only edit reuses the install layer.
COPY backend/pyproject.toml ./
COPY backend/app ./app
RUN pip install .

# Bake the committed knobs so the image is self-contained on a PaaS; compose
# mounts the live file over this in dev. The path is passed explicitly so
# nothing depends on the working directory.
COPY config.yaml ./config.yaml
ENV CONFIG_PATH=/srv/axiom/config.yaml

# Ingested corpora land here (bind-mounted in dev).
RUN mkdir -p /srv/axiom/data \
    && useradd --create-home --uid 10001 axiom \
    && chown -R axiom:axiom /srv/axiom
USER axiom

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
