FROM python:3.11-slim

# Install poetry
RUN pip install --no-cache-dir poetry
ENV POETRY_NO_INTERACTION=1

WORKDIR /app

# Copy pyproject.toml and poetry.lock for layer caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies (main only, no root package)
RUN poetry install --no-root --only main

# Copy the rest of the apps/api code
COPY apps/ ./apps/

# Expose port
EXPOSE 8000

# Note: sentence-transformers model download is slow on first container start.
# Mount a volume for the HF cache (~/.cache/huggingface) to speed up rebuilds:
#   volumes:
#     - hf_cache:/root/.cache/huggingface

# Run the application
CMD ["poetry", "run", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]