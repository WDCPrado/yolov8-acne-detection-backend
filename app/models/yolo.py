from ultralytics import YOLO


class YOLOModel:
    def __init__(self, model_path="app/weights/bestv1.pt"):
        self.model = YOLO(model_path)

    def predict(self, image, conf=0.5):
        results = self.model.predict(source=image, save=False, conf=conf)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": conf,
                        "class": cls,
                        "name": self.model.names[cls],
                    }
                )
        return detections
