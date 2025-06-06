# test_api.py
import requests

api_url = "http://127.0.0.1:8000/predict"
image_path = r"C:\Users\juanherj\OneDrive - Seguros Suramericana, S.A\Documentos\Esp Biociencias\API\brain_glioma_0001.jpg"

# Abrir el archivo en modo binario
with open(image_path, "rb") as f:
    files = {"image_file": f}
    response = requests.post(api_url, files=files)

try:
    response.raise_for_status()
    data = response.json()
    print("Respuesta del servidor:", data)
except requests.exceptions.HTTPError as errh:
    print("HTTP Error:", errh, response.text)
except requests.exceptions.RequestException as err:
    print("Error de conexión o de petición:", err)
