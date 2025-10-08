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

# Quick test: copy the entire repository into the image so committed model files
# (xgb_weather_model.pkl, xgb_scaler.pkl, xgb_label_encoder.pkl) are available
# at runtime. This is intended for quick verification only — for production prefer
# downloading models at build time from secure storage or using Git LFS.
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]