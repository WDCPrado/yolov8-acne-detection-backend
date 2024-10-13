from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
from io import BytesIO
import base64
from datetime import datetime
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
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from .detection import DetectionModel
from .data_models import PatientInfo, ExternalFactor, DetectionResult, AnalysisResult


class ExternalFactorsAnalyzer:
    def __init__(self, factor_weights: Dict[str, Dict[str, float]]):
        self.factor_weights = factor_weights

    def analyze(
        self, factors: List[ExternalFactor], detections: List[DetectionResult]
    ) -> Dict[str, float]:
        class_scores = {}
        class_counts = {}

        for detection in detections:
            class_name = detection.class_name
            confidence = detection.confidence
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += confidence

        if not detections:
            class_counts["Acné General"] = 1.0

        for class_name, total_confidence in class_counts.items():
            factor_weights = self.factor_weights.get(
                class_name, self.factor_weights["Acné General"]
            )
            class_score = 0
            for factor in factors:
                weight = factor_weights.get(factor.name, 0)
                class_score += factor.value * weight * total_confidence
            class_scores[class_name] = class_score

        return class_scores


class AcneAnalysisSystem:
    def __init__(
        self,
        detection_model: DetectionModel,
        external_factors_analyzer: ExternalFactorsAnalyzer,
    ):
        self.detection_model = detection_model
        self.external_factors_analyzer = external_factors_analyzer

    def analyze(
        self,
        image: Image.Image,
        factors: List[ExternalFactor],
        patient_info: PatientInfo,
    ) -> AnalysisResult:
        detections = self.detection_model.detect(image)
        factor_analysis = self.external_factors_analyzer.analyze(factors, detections)
        acne_type, severity = self.determine_acne_type_and_severity(
            factor_analysis, patient_info.age
        )
        recommendations = self.generate_recommendations(
            acne_type, severity, factors, patient_info
        )
        pdf_report = self.generate_pdf_report(
            image,
            detections,
            factor_analysis,
            acne_type,
            severity,
            recommendations,
            patient_info,
        )

        return AnalysisResult(
            detections=detections,
            factor_analysis=factor_analysis,
            acne_type=acne_type,
            severity=severity,
            recommendations=recommendations,
            pdf_report=pdf_report,
        )

    def determine_acne_type_and_severity(
        self, factor_analysis: Dict[str, float], age: int
    ) -> Tuple[str, str]:
        total_score = sum(factor_analysis.values())

        if age < 1:
            acne_type = "Acné Neonatal"
        elif 1 <= age < 7:
            acne_type = "Acné Infantil"
        elif 7 <= age < 25:
            acne_type = "Acné Vulgar"
        else:
            acne_type = "Acné Adulto"

        if total_score < 3:
            severity = "Leve"
        elif 3 <= total_score < 7:
            severity = "Moderado"
        else:
            severity = "Severo"

        return acne_type, severity

    def generate_recommendations(
        self,
        acne_type: str,
        severity: str,
        factors: List[ExternalFactor],
        patient_info: PatientInfo,
    ) -> List[str]:
        recommendations = []

        if severity == "Severo":
            recommendations.append(
                "Considere consultar a un dermatólogo para un tratamiento personalizado."
            )

        for factor in factors:
            if factor.name == "stress_level" and factor.value > 7:
                recommendations.append(
                    "Intente reducir el estrés mediante técnicas de relajación o meditación."
                )
            elif factor.name == "diet_quality" and factor.value < 5:
                recommendations.append(
                    "Mejore su dieta incluyendo más frutas, verduras y alimentos ricos en omega-3."
                )
            elif factor.name == "skin_type" and factor.value == 3:
                recommendations.append(
                    "Use productos no comedogénicos y limpie su rostro dos veces al día."
                )
            elif factor.name == "sun_exposure" and factor.value == 3:
                recommendations.append(
                    "Utilice protector solar diariamente para prevenir la inflamación y el daño cutáneo."
                )
            elif factor.name == "makeup_use" and factor.value == 3:
                recommendations.append(
                    "Opte por maquillaje no comedogénico y asegúrese de removerlo completamente antes de dormir."
                )

        if acne_type == "Acné Neonatal":
            recommendations.append(
                "El acné neonatal suele resolverse por sí solo. Mantenga la piel del bebé limpia y seca."
            )
        elif acne_type == "Acné Infantil":
            recommendations.append(
                "Consulte con un pediatra para determinar si se necesita tratamiento."
            )
        elif acne_type == "Acné Vulgar":
            recommendations.append(
                "Utilice productos de limpieza suaves y no irritantes. Evite tocar o apretar las lesiones."
            )
        elif acne_type == "Acné Adulto":
            recommendations.append(
                "Considere factores hormonales y de estrés. Un enfoque holístico puede ser beneficioso."
            )

        recommendations.append(
            "Mantenga una rutina de cuidado facial constante, usando productos adecuados para su tipo de piel."
        )
        return recommendations

    def generate_pdf_report(
        self,
        image: Image.Image,
        detections: List[DetectionResult],
        factor_analysis: Dict[str, float],
        acne_type: str,
        severity: str,
        recommendations: List[str],
        patient_info: PatientInfo,
    ) -> str:
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
        styles.add(ParagraphStyle(name="Justify", alignment=TA_JUSTIFY))
        styles.add(ParagraphStyle(name="Center", alignment=TA_CENTER))
        content = []

        # Portada
        content.append(Paragraph("Informe de Análisis de Acné", styles["Title"]))
        content.append(Spacer(1, 36))
        content.append(
            Paragraph(f"Preparado para: {patient_info.name}", styles["Heading2"])
        )
        content.append(Spacer(1, 12))
        content.append(
            Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", styles["Normal"])
        )
        content.append(PageBreak())

        # Información del Paciente
        content.append(Paragraph("Información del Paciente", styles["Heading1"]))
        patient_data = [
            ["Nombre", patient_info.name],
            ["Edad", str(patient_info.age)],
            [
                "Sexo",
                (
                    "Masculino"
                    if patient_info.sex == 0
                    else "Femenino" if patient_info.sex == 1 else "Otro"
                ),
            ],
        ]
        patient_table = Table(patient_data, colWidths=[2 * inch, 4 * inch])
        patient_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.beige),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        content.append(patient_table)
        content.append(Spacer(1, 12))

        # Diagnóstico
        content.append(Paragraph("Diagnóstico", styles["Heading1"]))
        content.append(Paragraph(f"Tipo de Acné: {acne_type}", styles["Heading2"]))
        content.append(Paragraph(f"Severidad: {severity}", styles["Heading2"]))
        content.append(Spacer(1, 12))

        # Información sobre el tipo de acné
        acne_info = {
            "Acné Neonatal": "El acné neonatal es una condición común que afecta a aproximadamente el 20% de los recién nacidos. Suele aparecer en las mejillas y generalmente se resuelve por sí solo en unas pocas semanas o meses.",
            "Acné Infantil": "El acné infantil es menos común y puede aparecer entre los 3 y 6 meses de edad. Puede requerir tratamiento si persiste o es severo.",
            "Acné Vulgar": "El acné vulgar es el tipo más común, afectando hasta el 85% de los adolescentes y jóvenes adultos. Es causado por una combinación de factores, incluyendo la producción excesiva de sebo, bacterias y inflamación.",
            "Acné Adulto": "El acné adulto afecta hasta el 15% de las mujeres y el 5% de los hombres mayores de 25 años. Puede ser causado por factores hormonales, estrés y ciertos productos para el cuidado de la piel.",
        }
        content.append(
            Paragraph(
                acne_info.get(
                    acne_type, "Información no disponible para este tipo de acné."
                ),
                styles["Justify"],
            )
        )
        content.append(Spacer(1, 12))

        # Análisis de Imagen
        content.append(Paragraph("Análisis de Imagen", styles["Heading1"]))
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        for detection in detections:
            x, y = detection.center
            r = 5  # Radio del círculo
            draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)

        img_buffer = BytesIO()
        img_copy.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        content.append(ReportLabImage(img_buffer, width=4 * inch, height=4 * inch))
        content.append(Spacer(1, 12))

        # Detecciones
        if detections:
            content.append(Paragraph("Lesiones Detectadas", styles["Heading2"]))
            detection_data = [["Tipo de Lesión", "Confianza"]]
            for detection in detections:
                detection_data.append(
                    [detection.class_name, f"{detection.confidence:.2f}"]
                )
            detection_table = Table(detection_data, colWidths=[3 * inch, 1 * inch])
            detection_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 1), (-1, -1), 10),
                        ("TOPPADDING", (0, 1), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            content.append(detection_table)
        else:
            content.append(
                Paragraph(
                    "No se detectaron lesiones específicas en la imagen.",
                    styles["Normal"],
                )
            )
        content.append(Spacer(1, 12))

        # Análisis de Factores
        content.append(Paragraph("Análisis de Factores", styles["Heading1"]))
        content.append(
            Paragraph(
                "Los siguientes factores pueden influir en la condición del acné:",
                styles["Normal"],
            )
        )
        factor_data = [["Factor", "Puntuación"]]
        for factor, score in factor_analysis.items():
            factor_data.append([factor, f"{score:.2f}"])
        factor_table = Table(factor_data, colWidths=[3 * inch, 1 * inch])
        factor_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 12),
                    ("TOPPADDING", (0, 1), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        content.append(factor_table)
        content.append(Spacer(1, 12))

        # Gráfico de factores
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 65
        pie.width = 130
        pie.height = 130
        pie.data = [score for score in factor_analysis.values()]
        pie.labels = [factor for factor in factor_analysis.keys()]
        pie.slices.strokeWidth = 0.5
        drawing.add(pie)
        content.append(drawing)
        content.append(Spacer(1, 12))

        # Explicación de factores
        factor_explanations = {
            "stress_level": "El estrés puede aumentar la producción de hormonas que estimulan las glándulas sebáceas, lo que puede empeorar el acné.",
            "diet_quality": "Una dieta rica en azúcares y grasas saturadas puede exacerbar el acné. Una dieta balanceada puede ayudar a reducir la inflamación.",
            "skin_type": "La piel grasa es más propensa al acné debido a la mayor producción de sebo.",
            "sun_exposure": "La exposición al sol puede inicialmente mejorar el acné, pero a largo plazo puede empeorar la condición y aumentar el riesgo de daño cutáneo.",
            "makeup_use": "El uso excesivo de maquillaje, especialmente productos no comedogénicos, puede obstruir los poros y empeorar el acné.",
        }
        for factor, explanation in factor_explanations.items():
            content.append(
                Paragraph(
                    f"<b>{factor.replace('_', ' ').title()}:</b> {explanation}",
                    styles["Justify"],
                )
            )
        content.append(Spacer(1, 12))

        # Recomendaciones
        content.append(Paragraph("Recomendaciones Personalizadas", styles["Heading1"]))
        for recommendation in recommendations:
            content.append(Paragraph(f"• {recommendation}", styles["BodyText"]))
        content.append(Spacer(1, 12))

        # Información adicional
        content.append(Paragraph("Información Adicional", styles["Heading1"]))
        content.append(
            Paragraph(
                "El acné es una condición común de la piel que afecta a millones de personas en todo el mundo. Aunque no es una condición grave, puede tener un impacto significativo en la autoestima y la calidad de vida. Es importante recordar que el acné es tratable y que existen muchas opciones disponibles para manejar esta condición.",
                styles["Justify"],
            )
        )
        content.append(Spacer(1, 12))
        content.append(
            Paragraph("Rutina de cuidado de la piel recomendada:", styles["Heading2"])
        )
        content.append(
            Paragraph("1. Limpieza suave dos veces al día", styles["BodyText"])
        )
        content.append(
            Paragraph("2. Uso de tónicos no alcohólicos", styles["BodyText"])
        )
        content.append(
            Paragraph(
                "3. Aplicación de tratamientos tópicos según lo recomendado",
                styles["BodyText"],
            )
        )
        content.append(
            Paragraph(
                "4. Hidratación con productos no comedogénicos", styles["BodyText"]
            )
        )
        content.append(Paragraph("5. Protección solar diaria", styles["BodyText"]))
        content.append(Spacer(1, 12))

        # Conclusión
        content.append(Paragraph("Conclusión", styles["Heading1"]))
        content.append(
            Paragraph(
                f"Basado en el análisis realizado, se ha determinado que usted tiene {acne_type} de severidad {severity.lower()}. Es importante seguir las recomendaciones proporcionadas y mantener una rutina de cuidado de la piel constante. Si los síntomas persisten o empeoran, se recomienda consultar a un dermatólogo para un tratamiento más específico.",
                styles["Justify"],
            )
        )
        content.append(Spacer(1, 12))

        # Disclaimer
        content.append(Paragraph("Aviso Legal", styles["Heading2"]))
        content.append(
            Paragraph(
                "Este informe es generado por un sistema de análisis automatizado y no sustituye el diagnóstico profesional de un dermatólogo. Siempre consulte a un profesional de la salud para obtener un diagnóstico y tratamiento personalizados.",
                styles["Justify"],
            )
        )

        doc.build(content)
        pdf_bytes = buffer.getvalue()
        return base64.b64encode(pdf_bytes).decode()
