import requests
import base64
from PIL import Image
import io
import os


def test_predict_endpoint(image_path, server_url="http://localhost:8000"):
    # Verificar si el archivo existe
    if not os.path.exists(image_path):
        print(f"Error: El archivo {image_path} no existe.")
        return

    # Leer la imagen y convertirla a base64
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error al leer la imagen: {e}")
        return

    # Preparar los datos para la solicitud
    data = {"image": encoded_string}

    # Hacer la solicitud POST al endpoint
    try:
        response = requests.post(f"{server_url}/predict", json=data)
        response.raise_for_status()  # Esto lanzará una excepción para códigos de error HTTP
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud: {e}")
        if response.text:
            print(f"Respuesta del servidor: {response.text}")
        return

    # Procesar la respuesta
    try:
        result = response.json()
    except ValueError:
        print("Error: No se pudo decodificar la respuesta JSON")
        print(f"Respuesta del servidor: {response.text}")
        return

    # Extraer la imagen resultante de base64
    try:
        image_data = base64.b64decode(result["image"])
        image = Image.open(io.BytesIO(image_data))

        # Guardar la imagen resultante
        output_path = "testing/resultado_prediccion.jpg"
        image.save(output_path)
        print(f"Imagen con predicciones guardada como: {output_path}")
    except Exception as e:
        print(f"Error al procesar la imagen resultante: {e}")

    # Imprimir las detecciones
    if "detections" in result:
        print("Detecciones:")
        for detection in result["detections"]:
            print(
                f"Clase: {detection['class_name']}, Confianza: {detection['confidence']:.2f}"
            )
    else:
        print("No se encontraron detecciones en la respuesta.")


if __name__ == "__main__":
    image_path = "testing/acne.png"
    test_predict_endpoint(image_path)
