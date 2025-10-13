FROM python:3.11-slim

WORKDIR /app

# Добавляем системные пакеты для PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc postgresql-client && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
