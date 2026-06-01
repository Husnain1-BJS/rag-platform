FROM python:3.11-slim

# Install Poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy pyproject.toml from project root
COPY pyproject.toml .

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the application
COPY apps/api/ .

# Expose port
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
