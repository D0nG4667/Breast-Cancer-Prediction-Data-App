# Use slim Python base
FROM python:3.12.10-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /src/client

# Copy lockfile and project metadata first (to leverage caching)
COPY uv.lock pyproject.toml ./

# Install dependencies only (non-editable, without project source)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable

# Copy project source
COPY . .

# Sync project into environment (non-editable)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# Final stage
FROM python:3.12.10-slim

WORKDIR /src/client

# Copy virtual environment from builder
COPY --from=builder /src/client/.venv /src/client/.venv
COPY . .

# Ensure venv binaries are on PATH
ENV PATH="/src/client/.venv/bin:$PATH"

# Expose app port
EXPOSE 8501

# Add healthcheck: curl the Streamlit server root
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start application
CMD ["uv", "run", "streamlit", "run", "app.py"]
