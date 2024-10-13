from ultralytics import YOLO
from PIL import Image
from typing import List
from .data_models import DetectionResult


class DetectionModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image) -> List[DetectionResult]:
        results = self.model(image)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = box.cls[0]
                class_name = self.model.names[int(class_id)]
                detections.append(
                    DetectionResult(
                        center=[(x1 + x2) / 2, (y1 + y2) / 2],
                        confidence=float(confidence),
                        class_name=class_name,
                    )
                )
        return detections
