# YOLO Acne Detection Backend

Este proyecto implementa un backend para la detección de acné utilizando YOLOv8, FastAPI y Docker. El sistema carga un modelo pre-entrenado para identificar y localizar acné en imágenes.

## Estructura del Proyecto

```
yolov8-acne-detection-backend
├── .dockerignore
├── .gitignore
├── app
│   ├── main.py
│   ├── weights
│   │   └── bestv1.pt
│   └── __pycache__
│       └── main.cpython-39.pyc
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
├── tasks.py
└── testing
    ├── acne.png
    ├── predict.py
    └── resultado_prediccion.jpg
```

- `app/main.py`: Contiene la lógica principal de la aplicación FastAPI.
- `app/weights/bestv1.pt`: Modelo pre-entrenado de YOLOv8 para la detección de acné.
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

## Uso del Modelo Pre-entrenado

El modelo YOLOv8 pre-entrenado para la detección de acné (`bestv1.pt`) se carga automáticamente al iniciar la aplicación. Este modelo ha sido entrenado específicamente para identificar y localizar áreas de acné en imágenes faciales.

## API Endpoints

- `POST /predict`: Acepta una imagen y devuelve las detecciones de acné.
  - Cuerpo de la solicitud: Archivo de imagen
  - Respuesta: JSON con las coordenadas de las detecciones y confianza

## Desarrollo

- Para actualizar dependencias:

  ```
  invoke update
  ```

- Para limpiar archivos temporales:
  ```
  invoke clean
  ```

## Notas Adicionales

- Asegúrate de que el modelo pre-entrenado (`bestv1.pt`) esté presente en la carpeta `app/weights/` antes de ejecutar la aplicación.
- La aplicación utiliza FastAPI, lo que permite una fácil extensión y documentación automática de la API.
- El uso de Invoke simplifica la ejecución de tareas comunes del proyecto.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de hacer un pull request.

## Licencia

[Incluir información de licencia aquí]
