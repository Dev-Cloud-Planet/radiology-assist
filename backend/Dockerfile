FROM python:3.11-slim

# Evitar buffering de logs y subir timeout de pip
ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

# Dependencias de sistema mínimas (para Pillow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# 1) Instalar PyTorch + Torchvision SOLO CPU (mucho más ligeros)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 2) Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
