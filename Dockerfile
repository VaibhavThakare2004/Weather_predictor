FROM python:3.10-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# application files only (models handled at runtime or via repo)
COPY api.py ./
COPY build.sh ./
COPY render.yaml ./

# Note: model files (xgb_*.pkl) are intentionally NOT copied here.
# If you want to include them in the image, add explicit COPY lines without shell operators.
# Recommended: host models in object storage and set MODEL_XGB_URL / MODEL_SCALER_URL / MODEL_LABEL_URL env vars in Render.

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]