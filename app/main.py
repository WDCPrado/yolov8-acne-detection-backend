from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import json
import os

from .models.data_models import PatientInfo, ExternalFactor, AnalysisResult
from .models.detection import DetectionModel
from .models.acne import ExternalFactorsAnalyzer, AcneAnalysisSystem

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your models and analysis system here
model_path = os.path.join(os.path.dirname(__file__), "models", "weights", "acne.pt")
detection_model = DetectionModel(model_path)

factor_weights = {
    "Acné General": {
        "stress_level": 0.25,
        "diet_quality": 0.25,
        "skin_type": 0.2,
        "sun_exposure": 0.15,
        "makeup_use": 0.15,
    },
    # Add other acne type-specific weights here if needed
}
external_factors_analyzer = ExternalFactorsAnalyzer(factor_weights)
acne_analysis_system = AcneAnalysisSystem(detection_model, external_factors_analyzer)


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(
    image: UploadFile = File(...),
    patient_info: str = Form(...),
    factors: str = Form(...),
):
    try:
        patient_info = PatientInfo(**json.loads(patient_info))
        factors = [ExternalFactor(**factor) for factor in json.loads(factors)]

        contents = await image.read()
        img = Image.open(BytesIO(contents))
        if img.mode != "RGB":
            img = img.convert("RGB")

        result = acne_analysis_system.analyze(
            img, factors, patient_info
        )  # Añadido patient_info aquí
        return result

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
