FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install dependencies first for fast layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy source packages and configs
COPY mars /app/mars
COPY interfaces /app/interfaces
COPY backend /app/backend
COPY configs /app/configs
COPY evals /app/evals

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
