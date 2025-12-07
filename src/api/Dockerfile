# Base image
FROM python:3.12.10-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /src/api

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

# Final stage
FROM python:3.12.10-slim

WORKDIR /src/api

# Copy virtual environment from builder
COPY --from=builder /src/api/.venv /src/api/.venv
COPY . .

# Ensure venv binaries are on PATH
ENV PATH="/src/api/.venv/bin:$PATH"

# Expose app port
EXPOSE 7860

# Add healthcheck: curl the API root or a /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Start application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
