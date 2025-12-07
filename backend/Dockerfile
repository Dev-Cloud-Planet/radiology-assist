# backend/Dockerfile (Versión 16: Fix de Paths y Cache)

FROM python:3.11-slim


LABEL org.opencontainers.image.source="https://github.com/Dev-Cloud-Planet/radiology-assist.git"
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

# Copiar requirements (La ruta es relativa a la raíz del proyecto: 'backend/requirements.txt')
COPY backend/requirements.txt . 

# 1) Instalar PyTorch + Torchvision SOLO CPU
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 2) Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE CACHE
# ----------------------------------------------------------------------
# Fuerza a torchxrayvision a usar una carpeta /tmp y asigna permisos 777.
ENV XRAY_DATA_DIR=/tmp/xrv-cache
RUN mkdir -p /tmp/xrv-cache && chmod 777 /tmp/xrv-cache

# ----------------------------------------------------------------------

# Copiar index.html (Asume que está en la raíz del proyecto)
COPY index.html .

# Copiar el código (La ruta es relativa a la raíz del proyecto: 'backend/app')
COPY backend/app ./app

EXPOSE 8000

# Comando final (Corregido desde V11)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]