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

        patient_info = {"name": "Paciente de Prueba", "age": 25, "sex": 0}

        factors = [
            {"name": "stress_level", "value": 4},
            {"name": "diet_quality", "value": 7},
            {"name": "skin_type", "value": 2},
            {"name": "sun_exposure", "value": 1},
            {"name": "makeup_use", "value": 1},
        ]

        data = {
            "patient_info": json.dumps(patient_info),
            "factors": json.dumps(factors),
        }

        try:
            response = requests.post(f"{server_url}/analyze", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            print("Análisis completado con éxito.")

            print("\nInformación del Paciente:")
            print(f"  Nombre: {patient_info['name']}")
            print(f"  Edad: {patient_info['age']}")
            print(
                f"  Sexo: {'Masculino' if patient_info['sex'] == 0 else 'Femenino' if patient_info['sex'] == 1 else 'Otro'}"
            )

            print("\nTipo de Acné:")
            print(f"  {result['acne_type']}")

            print("\nSeveridad:")
            print(f"  {result['severity']}")

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
            pdf_path = os.path.join(os.path.dirname(__file__), "informe_acne.pdf")
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(pdf_data)
            print(f"Informe PDF guardado como '{pdf_path}'")

        except requests.exceptions.RequestException as e:
            print(f"Error en la solicitud: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Respuesta del servidor: {e.response.text}")
        except json.JSONDecodeError:
            print("Error al decodificar la respuesta JSON del servidor.")
        except KeyError as e:
            print(f"Error: Falta la clave {e} en la respuesta del servidor.")
        except Exception as e:
            print(f"Error inesperado: {e}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "acne.png")
    test_analyze_endpoint(image_path)
