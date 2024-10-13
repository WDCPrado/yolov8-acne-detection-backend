from pydantic import BaseModel
from typing import List, Dict


class PatientInfo(BaseModel):
    name: str
    age: int
    sex: int  # 0: Masculino, 1: Femenino, 2: Otro


class ExternalFactor(BaseModel):
    name: str
    value: float


class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    center: List[float]


class AnalysisResult(BaseModel):
    detections: List[DetectionResult]
    factor_analysis: Dict[str, float]
    acne_type: str
    severity: str
    recommendations: List[str]
    pdf_report: str
