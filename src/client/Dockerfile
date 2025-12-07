# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

# Copy uv binary from Astral's official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create app directory
RUN mkdir /app
WORKDIR /app

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies (compilers, headers, pkg-config, MySQL client dev libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc pkg-config libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock /app/

# Install dependencies only (no project code yet)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy project code
ADD . /app

# Sync the project (install app into venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Precompile bytecode
ENV UV_COMPILE_BYTECODE=1


# ---- Stage 2: Runtime ----
FROM python:3.12-slim

# Copy uv binary again so runtime stage has it
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root user and app dir
RUN useradd -m -r appuser && \
    mkdir /app && \
    chown -R appuser /app

WORKDIR /app

# Copy app code + .venv together
COPY --from=builder --chown=appuser:appuser /app /app

# Install only runtime libraries (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure .venv/bin is first on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8501

# Silence CUDA and oneDNN warnings if not needed
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_ENABLE_ONEDNN_OPTS=0

# Healthcheck (adjust endpoint for your app)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start application with uv
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
