from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

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


class ImageUpload(BaseModel):
    image: str


def draw_prediction(img, x, y, class_name, confidence, img_width, img_height):
    draw = ImageDraw.Draw(img)

    # Calcular el tamaño del círculo basado en el tamaño de la imagen
    radius = int(min(img_width, img_height) * 0.02)  # Tamaño del círculo

    # Dibujar el círculo con un borde más delgado
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius], outline=(0, 255, 0), width=2
    )

    # Preparar el texto con confianza como porcentaje
    confidence_percentage = confidence * 100
    label = f"{class_name}: {confidence_percentage:.0f}%"  # Formato de porcentaje

    # Usar una fuente más grande y clara
    font_size = int(min(img_width, img_height) * 0.025)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Obtener el tamaño del texto
    text_width, text_height = draw.textsize(label, font=font)

    # Dibujar un rectángulo de fondo para el texto con bordes redondeados
    rect_coords = [
        x - text_width / 2 - 5,
        y - radius - text_height - 20,
        x + text_width / 2 + 5,
        y - radius - 10,
    ]
    draw.rounded_rectangle(rect_coords, radius=8, fill=(255, 255, 255, 200))

    # Dibujar el texto
    draw.text(
        (x - text_width / 2, y - radius - text_height - 15),
        label,
        fill=(0, 0, 0),
        font=font,
    )

    return img


@app.get("/")
async def root():
    return {"message": "Yolo Server is running"}


@app.post("/predict")
async def predict(body: ImageUpload):
    if not body.image:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        # Decodificar la imagen de base64
        img_data = base64.b64decode(body.image)
        img = Image.open(BytesIO(img_data))

        # Convertir la imagen a RGB si es necesario
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Guardar las dimensiones originales
        original_width, original_height = img.size

        # Redimensionar la imagen a 540x540 para la predicción
        img_resized = img.resize((540, 540))

        # Realizar la predicción
        results = model(img_resized)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = box.cls[0]
                class_name = model.names[int(class_id)]

                # Escalar las coordenadas de vuelta a la imagen original
                scale_x = original_width / 540
                scale_y = original_height / 540
                center_x = (x1 + x2) / 2 * scale_x
                center_y = (y1 + y2) / 2 * scale_y

                detections.append(
                    {
                        "center": [float(center_x), float(center_y)],
                        "confidence": float(confidence),
                        "class_id": int(class_id),
                        "class_name": class_name,
                    }
                )

                # Dibujar la predicción en la imagen original
                img = draw_prediction(
                    img,
                    center_x,
                    center_y,
                    class_name,
                    confidence,
                    original_width,
                    original_height,
                )

        # Convertir la imagen procesada a base64 con alta calidad
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"detections": detections, "image": img_base64}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
