# main.py

import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from huggingface_hub import hf_hub_download
from crop_utils import CropBlackBorders

# ------------------------------------------------------------
# 3.2. Instanciar FastAPI y configurar dispositivo
# ------------------------------------------------------------
app = FastAPI(title="API de Clasificación de Tumores Cerebrales")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# 3.3. Función para cargar el modelo de imagen desde HuggingFace
# ------------------------------------------------------------
def load_image_model():
    """
    Descarga y carga EfficientNet-B0 afinado para 3 clases de tumor cerebral.
    Retorna el modelo listo para inferencia en GPU/CPU.
    """
    repo_id = "jherronr/efficientnet-brain-tumor-classifier"
    filename = "pytorch_model.bin" # Nombre del archivo de pesos en el repo

    try:
        # 3.3.1. Descargar el archivo de pesos desde HuggingFace Hub
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        raise RuntimeError(f"Error descargando el modelo: {e}")

    # 3.3.2. Instanciar EfficientNet-B0 sin preentrenamiento en ImageNet
    model = models.efficientnet_b0(pretrained=False)

    # 3.3.3. Ajustar la última capa a 3 salidas (glioma, meningioma, tumor genérico)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 3)

    try:
        # 3.3.4. Cargar el diccionario de pesos en el modelo
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error cargando pesos en el modelo: {e}")

    model.to(device)
    model.eval()
    return model

# Cargar el modelo SOLO UNA VEZ al iniciar la app
image_model = load_image_model()

# ------------------------------------------------------------
# 3.4. Definir el pipeline de transformaciones (incluye CropBlackBorders)
# ------------------------------------------------------------
image_transform = transforms.Compose([
    CropBlackBorders(),                 # 1. Recortar bordes negros
    transforms.Resize((224, 224)),      # 2. Redimensionar a 224×224
    transforms.ToTensor(),              # 3. Convertir a tensor [0..1]
    transforms.Normalize([0.5, 0.5, 0.5],     # 4. Normalizar cada canal a [-1..1]
                         [0.5, 0.5, 0.5])
])

# Mapeo de índice → etiqueta (igual que en entrenamiento)
idx2label_image = {
    0: "brain_glioma",
    1: "brain_menin",
    2: "brain_tumor"
}

# ------------------------------------------------------------
# 3.5. Endpoint /predict para recibir la imagen y devolver la predicción
# ------------------------------------------------------------
@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    """
    Recibe:
      - image_file: archivo .jpg/.png con MRI de tumor cerebral.
    Devuelve:
      - tumor_prediction: etiqueta predicha (brain_glioma, brain_menin, brain_tumor).
    """
    # 3.5.1. Validar formato del archivo
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=415,
            detail="Formato de archivo no soportado. Use JPG o PNG."
        )

    # 3.5.2. Leer la imagen subida en memoria
    try:
        contents = await image_file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al leer la imagen: {e}"
        )

    # 3.5.3. Preprocesar la imagen (CropBlackBorders + Resize + ToTensor + Normalize)
    try:
        img_cropped = image_transform(pil_image)   # Pipeline completo 
        img_tensor = img_cropped.unsqueeze(0).to(device)  # Añadir dim batch: [1, 1, 224, 224]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en preprocesamiento de imagen: {e}"
        )

    # 3.5.4. Inferencia con el modelo
    try:
        with torch.no_grad():
            outputs = image_model(img_tensor)         # logits [1,3]
            _, preds = torch.max(outputs, dim=1)      # índice de clase
            idx = preds.item()
            tumor_label = idx2label_image[idx]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en inferencia del modelo: {e}"
        )

    # 3.5.5. Armado de la respuesta JSON
    response = {
        "tumor_prediction": tumor_label
    }
    return JSONResponse(content=response)


