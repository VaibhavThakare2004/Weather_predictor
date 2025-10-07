# Dockerfile for FastAPI backend
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py ./
# Copy model files if present - optional
COPY xgb_weather_model.pkl ./ || true
COPY xgb_scaler.pkl ./ || true
COPY xgb_label_encoder.pkl ./ || true

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
