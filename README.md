# Sistema de Análisis y Detección de Acné

Este proyecto implementa un backend completo para la detección y análisis de acné utilizando YOLOv8, FastAPI y Docker. El sistema carga un modelo pre-entrenado para identificar y localizar acné en imágenes, analiza factores externos y genera un informe detallado en PDF.

## Características Principales

- Detección de acné en imágenes utilizando YOLOv8
- Análisis de factores externos que influyen en el acné
- Generación de recomendaciones personalizadas
- Creación de informes detallados en PDF
- API RESTful para fácil integración

## Estructura del Proyecto

```
yolov8-acne-detection-backend
├── .dockerignore
├── .gitignore
├── app
│   ├── fonts
│   │   └── Roboto
│   │       ├── LICENSE.txt
│   │       ├── Roboto-*.ttf
│   ├── main.py
│   ├── watermark.png
│   ├── weights
│   │   └── bestv1.pt
│   └── __pycache__
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
├── tasks.py
└── testing
    ├── acne.png
    ├── informe_acne.pdf
    ├── predict.py
    └── resultado_prediccion.jpg
```

- `app/main.py`: Lógica principal de la aplicación FastAPI.
- `app/weights/bestv1.pt`: Modelo pre-entrenado de YOLOv8 para la detección de acné.
- `app/fonts/Roboto`: Fuentes utilizadas en la generación de PDF.
- `app/watermark.png`: Imagen de marca de agua para los informes PDF.
- `docker-compose.yml`: Configuración para ejecutar la aplicación en Docker.
- `Dockerfile`: Instrucciones para construir la imagen Docker.
- `requirements.txt`: Lista de dependencias de Python.
- `tasks.py`: Tareas de Invoke para facilitar la ejecución de comandos comunes.

## Requisitos Previos

- Python 3.9+
- Conda (recomendado para gestión de entornos)
- Docker y Docker Compose (para ejecución en contenedores)

## Configuración del Entorno

1. Crea un entorno Conda:

   ```
   conda create -n yolo-env python=3.9
   conda activate yolo-env
   ```

2. Instala Invoke:

   ```
   pip install invoke
   ```

3. Instala las dependencias del proyecto:
   ```
   invoke install
   ```

## Ejecución del Proyecto

### Usando Invoke

- Iniciar en modo desarrollo:

  ```
  invoke dev
  ```

- Iniciar en modo producción:
  ```
  invoke start
  ```

### Usando Docker

1. Construye la imagen:

   ```
   docker-compose build
   ```

2. Inicia el contenedor:
   ```
   docker-compose up
   ```

## API Endpoints

- `POST /analyze`: Analiza una imagen de acné y proporciona un informe detallado.
  - Cuerpo de la solicitud:
    - `image`: Archivo de imagen
    - `patient_info`: Información del paciente (JSON)
    - `factors`: Factores externos (JSON)
  - Respuesta: JSON con detecciones, análisis de factores, recomendaciones y un informe PDF codificado en base64.

## Desarrollo

- Para actualizar dependencias:

  ```
  invoke update
  ```

- Para limpiar archivos temporales:

  ```
  invoke clean
  ```

- Para ejecutar una predicción de prueba:
  ```
  invoke predict
  ```

## Notas Adicionales

- Asegúrate de que el modelo pre-entrenado (`bestv1.pt`) esté presente en la carpeta `app/weights/` antes de ejecutar la aplicación.
- La aplicación utiliza FastAPI, lo que permite una fácil extensión y documentación automática de la API.
- El sistema genera informes detallados en PDF utilizando ReportLab.
- Se incluye un análisis de factores externos que influyen en el acné.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de hacer un pull request. test

## Licencia

Este proyecto está licenciado bajo una Licencia MIT Modificada - ver el archivo [LICENSE](LICENSE) para más detalles.
