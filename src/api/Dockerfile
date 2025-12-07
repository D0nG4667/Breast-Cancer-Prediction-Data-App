# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

# Copy uv binary from Astral's official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /src/api

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy lockfile and project metadata first (to leverage caching)
COPY uv.lock pyproject.toml ./

# Install dependencies only (non-editable, without project source)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable

# Copy API source
COPY . .

# Sync project into environment (non-editable)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable


# ---- Stage 2: Runtime ----
FROM python:3.12-slim

# Copy uv binary again so runtime stage has it
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /src/api

# Copy virtual environment and app code from builder
COPY --from=builder /src/api/.venv /src/api/.venv
COPY --from=builder /src/api /src/api

# Ensure venv binaries are on PATH
ENV PATH="/src/api/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose app port
EXPOSE 7860

# Silence CUDA and oneDNN warnings if not needed
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_ENABLE_ONEDNN_OPTS=0


# Add healthcheck: curl the API root or a /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Start application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
