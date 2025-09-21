# Base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy uv binaries from official uv image
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

# Install system dependencies and clean up
RUN apt-get update -y \
    && apt-get install -y awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and uv.lock from root (root is build context)
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --locked

# Copy only ML app code
COPY agent/ ./agent/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI using uv
CMD ["uv", "run", "uvicorn", "agent.app:app", "--host", "0.0.0.0", "--port", "8000"]