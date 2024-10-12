from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from ultralytics import YOLO
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import json
import os
import math

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as ReportLabImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import Color

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("app/weights/bestv1.pt")

# Registrar fuentes personalizadas
pdfmetrics.registerFont(TTFont("Roboto-Regular", "app/fonts/Roboto/Roboto-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Roboto-Bold", "app/fonts/Roboto/Roboto-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Roboto-Italic", "app/fonts/Roboto/Roboto-Italic.ttf"))


class PatientInfo(BaseModel):
    name: str
    age: int
    sex: int  # 0: Masculino, 1: Femenino, 2: Otro


class ExternalFactors(BaseModel):
    stress_level: int  # Nivel de estrés (1-10)
    diet_quality: int  # Calidad de la dieta (1-10)
    skin_type: int  # Tipo de piel (1: Seca, 2: Mixta, 3: Grasosa, 4: Sensible)
    sun_exposure: int  # Exposición al sol (1: Baja, 2: Moderada, 3: Alta)
    makeup_use: int  # Uso de maquillaje (1: Rara vez, 2: A veces, 3: Frecuente)


class AcneAnalysisResult(BaseModel):
    detections: List[Dict]
    factor_analysis: Dict[str, float]
    recommendations: List[str]
    pdf_report: str


class AcneAnalysisSystem:
    def __init__(self):
        self.ACNE_TYPE_FACTOR_WEIGHTS = {
            "Acné General": {
                "stress_level": 0.25,
                "diet_quality": 0.25,
                "skin_type": 0.2,
                "sun_exposure": 0.15,
                "makeup_use": 0.15,
            },
            "Acné Comedonal": {
                "stress_level": 0.2,
                "diet_quality": 0.3,
                "skin_type": 0.3,
                "sun_exposure": 0.1,
                "makeup_use": 0.1,
            },
            "Acné Inflamatorio": {
                "stress_level": 0.3,
                "diet_quality": 0.2,
                "skin_type": 0.25,
                "sun_exposure": 0.15,
                "makeup_use": 0.1,
            },
            "Acné Quístico": {
                "stress_level": 0.35,
                "diet_quality": 0.15,
                "skin_type": 0.2,
                "sun_exposure": 0.2,
                "makeup_use": 0.1,
            },
        }

    def analyze_external_factors(
        self, factors: ExternalFactors, detections: List[Dict]
    ) -> Dict[str, float]:
        acne_scores = {}
        acne_type_counts = {}
        for detection in detections:
            acne_type = detection["class_name"]
            confidence = detection["confidence"]
            if acne_type not in acne_type_counts:
                acne_type_counts[acne_type] = 0
            acne_type_counts[acne_type] += confidence

        if not detections:
            acne_type_counts["Acné General"] = 1.0

        for acne_type, total_confidence in acne_type_counts.items():
            factor_weights = self.ACNE_TYPE_FACTOR_WEIGHTS.get(
                acne_type, self.ACNE_TYPE_FACTOR_WEIGHTS["Acné General"]
            )
            acne_score = 0
            for factor, weight in factor_weights.items():
                factor_value = getattr(factors, factor)
                normalized_value = self.normalize_factor(factor, factor_value)
                acne_score += normalized_value * weight * total_confidence
            acne_scores[acne_type] = acne_score
        return acne_scores

    def normalize_factor(self, factor: str, value: int) -> float:
        if factor in ["stress_level", "diet_quality"]:
            return value / 10
        elif factor == "skin_type":
            return {1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9}.get(value, 0.5)
        elif factor == "sun_exposure":
            return {1: 0.3, 2: 0.6, 3: 0.9}.get(value, 0.5)
        elif factor == "makeup_use":
            return {1: 0.3, 2: 0.6, 3: 0.9}.get(value, 0.5)
        else:
            return 0.5

    def generate_recommendations(
        self, factor_analysis: Dict[str, float], factors: ExternalFactors
    ) -> List[str]:
        recommendations = []
        total_acne_score = sum(factor_analysis.values())

        if total_acne_score > 7:
            recommendations.append(
                "Considere consultar a un dermatólogo para un tratamiento personalizado."
            )
        if factors.stress_level > 7:
            recommendations.append(
                "Intente reducir el estrés mediante técnicas de relajación o meditación."
            )
        if factors.diet_quality < 5:
            recommendations.append(
                "Mejore su dieta incluyendo más frutas, verduras y alimentos ricos en omega-3."
            )
        if factors.skin_type == 3:
            recommendations.append(
                "Use productos no comedogénicos y limpie su rostro dos veces al día."
            )
        if factors.sun_exposure == 3:
            recommendations.append(
                "Utilice protector solar diariamente para prevenir la inflamación y el daño cutáneo."
            )
        if factors.makeup_use == 3:
            recommendations.append(
                "Opte por maquillaje no comedogénico y asegúrese de removerlo completamente antes de dormir."
            )

        recommendations.append(
            "Mantenga una rutina de cuidado facial constante, usando productos adecuados para su tipo de piel."
        )
        return recommendations

    def analyze_image(self, image: Image.Image) -> List[Dict]:
        results = model(image)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = box.cls[0]
                class_name = model.names[int(class_id)]
                class_name = self.map_class_name(class_name)
                detections.append(
                    {
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                        "confidence": float(confidence),
                        "class_id": int(class_id),
                        "class_name": class_name,
                    }
                )
        return detections

    def map_class_name(self, class_name: str) -> str:
        class_mapping = {
            "acne": "Acné General",
            "blackhead": "Acné Comedonal",
            "whitehead": "Acné Comedonal",
            "papule": "Acné Inflamatorio",
            "pustule": "Acné Inflamatorio",
            "nodule": "Acné Quístico",
            "cyst": "Acné Quístico",
        }
        return class_mapping.get(class_name.lower(), "Acné General")


def create_acne_report(
    patient_info: PatientInfo,
    acne_types: List[str],
    predicted_image: bytes,
    factor_analysis: Dict[str, float],
    recommendations: List[str],
    detections: List[Dict],
    factors: ExternalFactors,
) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()

    styles["Title"].fontName = "Roboto-Bold"
    styles["Title"].fontSize = 18
    styles["Title"].spaceAfter = 12
    styles["Title"].alignment = TA_CENTER

    styles["Heading1"].fontName = "Roboto-Bold"
    styles["Heading1"].fontSize = 16
    styles["Heading1"].spaceAfter = 12

    styles["Heading2"].fontName = "Roboto-Bold"
    styles["Heading2"].fontSize = 14
    styles["Heading2"].spaceAfter = 12

    styles["Normal"].fontName = "Roboto-Regular"
    styles["Normal"].fontSize = 11
    styles["Normal"].leading = 14

    styles.add(
        ParagraphStyle(name="Justify", parent=styles["Normal"], alignment=TA_JUSTIFY)
    )

    content = []
    content.append(Paragraph("Informe de Análisis de Acné", styles["Title"]))
    content.append(Spacer(1, 12))

    patient_table_data = [
        [
            Paragraph("Nombre del Paciente:", styles["Normal"]),
            Paragraph(patient_info.name, styles["Normal"]),
        ],
        [
            Paragraph("Edad:", styles["Normal"]),
            Paragraph(f"{patient_info.age} años", styles["Normal"]),
        ],
        [
            Paragraph("Sexo:", styles["Normal"]),
            Paragraph(
                f"{'Masculino' if patient_info.sex == 0 else 'Femenino' if patient_info.sex == 1 else 'Otro'}",
                styles["Normal"],
            ),
        ],
    ]
    patient_table = Table(patient_table_data, colWidths=[2.5 * inch, 3.5 * inch])
    patient_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    content.append(patient_table)
    content.append(Spacer(1, 12))

    img_width = 4 * inch
    img_height = 4 * inch
    img = Image.open(BytesIO(predicted_image))
    img_buffer = BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    content.append(Paragraph("Resultado del Análisis de Imagen", styles["Heading2"]))
    content.append(ReportLabImage(img_buffer, width=img_width, height=img_height))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Análisis de la Imagen", styles["Heading2"]))
    types_list = ", ".join(set(acne_types))
    content.append(
        Paragraph(
            f"Los tipos de acné identificados en su caso son: <b>{types_list}</b>.",
            styles["Normal"],
        )
    )
    content.append(Spacer(1, 12))

    content.append(Paragraph("Análisis de Factores Externos", styles["Heading2"]))
    factor_explanations = {
        "stress_level": "El estrés puede aumentar la producción de sebo, lo que puede empeorar el acné.",
        "diet_quality": "Una dieta balanceada ayuda a reducir la inflamación y mejora la salud de la piel.",
        "skin_type": "El tipo de piel influye en la producción de sebo y la tendencia a desarrollar acné.",
        "sun_exposure": "La exposición al sol puede dañar la piel y empeorar las cicatrices de acné.",
        "makeup_use": "El uso frecuente de maquillaje puede obstruir los poros y agravar el acné.",
    }

    skin_type_mapping = {1: "Seca", 2: "Mixta", 3: "Grasosa", 4: "Sensible"}
    sun_exposure_mapping = {1: "Baja", 2: "Moderada", 3: "Alta"}
    makeup_use_mapping = {1: "Rara vez", 2: "A veces", 3: "Frecuente"}

    for factor in factor_explanations.keys():
        factor_value = getattr(factors, factor)
        readable_value = factor_value
        if factor == "skin_type":
            readable_value = skin_type_mapping.get(factor_value, "Desconocido")
        elif factor == "sun_exposure":
            readable_value = sun_exposure_mapping.get(factor_value, "Desconocido")
        elif factor == "makeup_use":
            readable_value = makeup_use_mapping.get(factor_value, "Desconocido")
        elif factor in ["stress_level", "diet_quality"]:
            readable_value = f"{factor_value}/10"
        content.append(
            Paragraph(
                f"<b>{factor.replace('_', ' ').title()}:</b> {readable_value}",
                styles["Normal"],
            )
        )
        content.append(Paragraph(factor_explanations[factor], styles["Justify"]))
        content.append(Spacer(1, 6))

    content.append(Spacer(1, 12))

    content.append(Paragraph("Interpretación General", styles["Heading2"]))
    content.append(
        Paragraph(
            "Basado en nuestro análisis, se han identificado los siguientes puntajes para cada tipo de acné:",
            styles["Justify"],
        )
    )
    content.append(Spacer(1, 12))

    acne_scores_table_data = [
        [
            Paragraph("<b>Tipo de Acné</b>", styles["Normal"]),
            Paragraph("<b>Puntaje</b>", styles["Normal"]),
        ]
    ]
    for acne_type, score in factor_analysis.items():
        acne_scores_table_data.append(
            [
                Paragraph(acne_type, styles["Normal"]),
                Paragraph(f"{score:.2f}", styles["Normal"]),
            ]
        )
    acne_scores_table = Table(acne_scores_table_data, colWidths=[3 * inch, 1.5 * inch])
    acne_scores_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Roboto-Regular"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    content.append(acne_scores_table)
    content.append(Spacer(1, 12))

    # Añadir explicación de la escala de puntaje
    content.append(Paragraph("Escala de Puntaje de Acné:", styles["Heading2"]))
    content.append(Paragraph("0-3: Acné leve", styles["Normal"]))
    content.append(Paragraph("3-7: Acné moderado", styles["Normal"]))
    content.append(Paragraph("7-10: Acné severo", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Recomendaciones Personalizadas", styles["Heading2"]))
    for rec in recommendations:
        content.append(Paragraph(f"• {rec}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            "Es fundamental seguir una rutina de cuidado de la piel adecuada y considerar las recomendaciones proporcionadas. "
            "Si los síntomas persisten o se agravan, consulte a un dermatólogo para obtener un tratamiento especializado.",
            styles["Justify"],
        )
    )
    content.append(Spacer(1, 12))

    def add_watermark(canvas, doc):
        canvas.saveState()
        if os.path.exists("app/watermark.png"):
            watermark = Image.open("app/watermark.png")
            watermark_width, watermark_height = watermark.size
            page_width, page_height = letter

            # Calcular el número de repeticiones necesarias
            repeats_x = math.ceil(page_width / watermark_width) + 1
            repeats_y = math.ceil(page_height / watermark_height) + 1

            for i in range(repeats_x):
                for j in range(repeats_y):
                    x = i * watermark_width - (watermark_width / 2)
                    y = j * watermark_height - (watermark_height / 2)
                    canvas.setFillAlpha(0.1)
                    canvas.drawImage(
                        "app/watermark.png",
                        x,
                        y,
                        width=watermark_width,
                        height=watermark_height,
                        mask="auto",
                    )
                    canvas.setFillAlpha(1)

        canvas.restoreState()
        page_num = canvas.getPageNumber()
        text = f"Página {page_num}"
        canvas.setFont("Roboto-Regular", 9)
        canvas.drawRightString(letter[0] - 72, 15 * mm, text)

    doc.build(content, onFirstPage=add_watermark, onLaterPages=add_watermark)
    buffer.seek(0)
    return buffer


acne_system = AcneAnalysisSystem()


@app.post("/analyze", response_model=AcneAnalysisResult)
async def analyze_acne(
    image: UploadFile = File(...),
    patient_info: str = Form(...),
    factors: str = Form(...),
):
    try:
        patient_info = PatientInfo(**json.loads(patient_info))
        factors = ExternalFactors(**json.loads(factors))

        contents = await image.read()
        img = Image.open(BytesIO(contents))
        if img.mode != "RGB":
            img = img.convert("RGB")

        detections = acne_system.analyze_image(img)

        draw = ImageDraw.Draw(img)
        for detection in detections:
            x, y = detection["center"]
            r = 5  # Radio del círculo
            draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        processed_image = img_buffer.getvalue()

        factor_analysis = acne_system.analyze_external_factors(factors, detections)
        recommendations = acne_system.generate_recommendations(factor_analysis, factors)

        acne_types_detected = [detection["class_name"] for detection in detections]
        if not acne_types_detected:
            acne_types_detected = ["Acné General"]

        pdf_buffer = create_acne_report(
            patient_info,
            acne_types_detected,
            processed_image,
            factor_analysis,
            recommendations,
            detections,
            factors,
        )
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode()

        return AcneAnalysisResult(
            detections=detections,
            factor_analysis=factor_analysis,
            recommendations=recommendations,
            pdf_report=pdf_base64,
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
