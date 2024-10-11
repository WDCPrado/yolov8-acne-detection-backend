# Usa una imagen base de Python
FROM python:3.9-slim-buster

# Instala curl y tzdata
RUN apt-get update && apt-get install -y curl tzdata

# Establece la zona horaria de Chile
ENV TZ=America/Santiago

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requerimientos e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de la aplicación
COPY ./app ./app

# Expone el puerto que la aplicación escuchará
EXPOSE 80

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]