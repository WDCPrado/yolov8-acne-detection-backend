from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import base64
from skimage import io, draw, color
from skimage.transform import resize
from skimage.util import img_as_ubyte
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo YOLO
model = YOLO("app/weights/bestv1.pt")


@app.get("/")
async def root():
    return {"message": "Yolo Server is running"}


def draw_prediction(img, x, y, class_name, confidence):
    # Dibujar el círculo
    radius = 20
    rr, cc = draw.circle_perimeter(int(y), int(x), radius)
    img[rr, cc] = [0, 255, 0]  # Color verde

    # Preparar el texto
    label = f"{class_name}: {confidence:.2f}"

    # Convertir la imagen a PIL para dibujar texto
    pil_img = Image.fromarray(img_as_ubyte(img))
    draw = ImageDraw.Draw(pil_img)

    # Usar una fuente por defecto
    font = ImageFont.load_default()

    # Obtener el tamaño del texto
    text_width, text_height = draw.textsize(label, font=font)

    # Dibujar un rectángulo de fondo para el texto
    draw.rectangle(
        [
            int(x - text_width / 2),
            int(y - radius - text_height - 5),
            int(x + text_width / 2),
            int(y - radius),
        ],
        fill=(255, 255, 255),
    )

    # Dibujar el texto
    draw.text(
        (int(x - text_width / 2), int(y - radius - text_height - 5)),
        label,
        fill=(0, 0, 0),
        font=font,
    )

    # Convertir de vuelta a numpy array
    return np.array(pil_img)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = io.imread(BytesIO(contents))

    # Asegurarse de que la imagen esté en RGB
    if img.shape[-1] == 4:  # Si es RGBA
        img = color.rgba2rgb(img)

    results = model(img)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = model.names[int(class_id)]

            # Calcular el centro del bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detections.append(
                {
                    "center": [float(center_x), float(center_y)],
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": class_name,
                }
            )

            # Dibujar la predicción en la imagen
            img = draw_prediction(img, center_x, center_y, class_name, confidence)

    # Convertir la imagen a bytes
    img_pil = Image.fromarray(img_as_ubyte(img))
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"detections": detections, "image": img_base64}
