import requests
import json
import os
import base64


def test_analyze_endpoint(image_path, server_url="http://localhost:8000"):
    if not os.path.exists(image_path):
        print(f"Error: El archivo {image_path} no existe.")
        return

    with open(image_path, "rb") as image_file:
        files = {"image": ("image.jpg", image_file, "image/jpeg")}
        data = {
            "patient_info": json.dumps(
                {"name": "Paciente de Prueba", "age": 25, "sex": 0}
            ),
            "factors": json.dumps(
                {
                    "stress_level": 7,
                    "diet_quality": 6,
                    "skin_type": 2,
                    "sun_exposure": 2,
                    "makeup_use": 1,
                }
            ),
        }

        try:
            response = requests.post(f"{server_url}/analyze", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            print("Análisis completado con éxito.")
            print("\nDetecciones:")
            for detection in result["detections"]:
                print(
                    f"  Clase: {detection['class_name']}, Confianza: {detection['confidence']:.2f}"
                )

            print("\nAnálisis de Factores:")
            for factor, score in result["factor_analysis"].items():
                print(f"  {factor}: {score:.2f}")

            print("\nRecomendaciones:")
            for recommendation in result["recommendations"]:
                print(f"  - {recommendation}")

            print("\nInforme PDF generado y disponible en la respuesta.")

            # Guardar el PDF
            pdf_data = base64.b64decode(result["pdf_report"])
            with open("testing/informe_acne.pdf", "wb") as pdf_file:
                pdf_file.write(pdf_data)
            print("Informe PDF guardado como 'informe_acne.pdf'")

        except requests.exceptions.RequestException as e:
            print(f"Error en la solicitud: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Respuesta del servidor: {e.response.text}")


if __name__ == "__main__":
    image_path = "testing/acne.png"
    test_analyze_endpoint(image_path)
