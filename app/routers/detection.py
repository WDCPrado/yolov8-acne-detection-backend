from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.models.yolo import YOLOModel
from app.utils.image_processing import process_image, draw_boxes
import base64

router = APIRouter()
model = YOLOModel()


@router.post("/detect_image")
async def detect(file: UploadFile = File(...)):
    try:
        image = await process_image(file)
        detections = model.predict(image)
        image_with_boxes = draw_boxes(image, detections)

        # Encode the image to base64
        _, buffer = cv2.imencode(
            ".jpg", image_with_boxes, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        return {"image": jpg_as_text, "detections": detections}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
