from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las fuentes
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados HTTP
)

# Cargar el modelo YOLO preentrenado
model = YOLO("models/bestv1.pt")

@app.get("/")
async def root():
    return {"message": "Servidor Yolo abierto"}

@app.post("/detect_image")
async def detect(file: UploadFile = File(...)):
    try:
        # Leer el archivo cargado
        contents = await file.read()
        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                content={"error": "Error: Cannot decode image"}, status_code=400
            )

        # Realizar la predicción
        results = model.predict(source=image, save=False, conf=0.5)

        # Extraer datos de predicción
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                    "class": cls,
                    "name": model.names[cls]
                })

                # Dibujar las cajas en la imagen
                color = (221, 160, 221)
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Tamaño del texto
                    color,
                    2,  # Borde del texto
                )

        # Encode the image back to base64 with high resolution
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        return {"image": jpg_as_text, "detections": detections}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


