FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install essential system build dependencies and OpenMP for XGBoost/LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Default launch command: FastAPI serving service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
