# Usa una imagen base de Python
FROM python:3.9-slim-buster

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0

# Establece la zona horaria de Chile
ENV TZ=America/Santiago

# Establece el directorio de trabajo
WORKDIR /acne-detection-backend

# Copia los archivos de requerimientos e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de la aplicación
COPY . .

# Expone el puerto que la aplicación escuchará
EXPOSE 80

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]