# Usar Python 3.10 slim para optimizar el tamaño de imagen
FROM python:3.10-slim

# Evitar la escritura de bytecode de python y buffering de salida
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema requeridas por OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libgles2 \
    libopengl0 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Descargar los modelos oficiales de MediaPipe Tasks
RUN curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task \
    && curl -O https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite

# Copiar e instalar dependencias de Python (torch CPU para reducir tamaño)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==2.6.0+cpu" \
    && pip install --no-cache-dir -r requirements.txt

# Descargar los pesos del modelo Lightweight 3D
RUN gdown 1niBUbUecPhKt3GyeDNukobL4OQ3jqssH -O human-pose-estimation-3d.pth

# Copiar el código de la aplicación y los tests
COPY pytest.ini .
COPY app/ ./app/
COPY tests/ ./tests/

# Exponer el puerto del servicio REST API
EXPOSE 8000

# Comando para ejecutar la aplicación con Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
