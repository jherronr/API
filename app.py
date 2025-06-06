# app.py

import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from crop_utils import CropBlackBorders

# ------------------------------------------------
# 1.2. Instanciar FastAPI y configurar dispositivo
# ------------------------------------------------
app = FastAPI(title="API: Clasificación de Tumor y Tratamiento")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# 1.3. Función para cargar modelo de imagen
# ------------------------------------------------
def load_image_model():
    """
    Descarga y carga EfficientNet-B0 afinado para 3 clases de tumor.
    """
    repo_id = "jherronr/efficientnet-brain-tumor-classifier"
    filename = "pytorch_model.bin"

    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        raise RuntimeError(f"Error descargando el modelo de imagen: {e}")

    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 3)

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error cargando pesos en el modelo de imagen: {e}")

    model.to(device)
    model.eval()
    return model


# Cargar el modelo de imagen UNA VEZ
image_model = load_image_model()

# ------------------------------------------------
# 1.4. Pipeline de transformaciones para imagen RGB
# ------------------------------------------------
image_transform = transforms.Compose([
    CropBlackBorders(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Índice → etiqueta de tumor
idx2label_image = {
    0: "Brain Glioma",
    1: "Brain Meningiomas",
    2: "Brain Tumor"
}


# ------------------------------------------------
# 1.5. Función para cargar modelo de texto (ClinicalBERT)
# ------------------------------------------------
def load_text_model():
    """
    Descarga y carga ClinicalBERT afinado para clasificación de tratamiento.
    El repo en HF debe contener config (con id2label), tokenizer y pesos.
    """
    repo_id = "jherronr/clinicalbert-treatment-classifier"

    try:
        text_model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo de texto de HuggingFace: {e}")

    text_model.to(device)
    text_model.eval()

    text_tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return text_model, text_tokenizer


# Cargar el modelo de texto UNA VEZ
text_model, text_tokenizer = load_text_model()

# Índice → etiqueta de tratamiento (desde config.id2label)
idx2label_text = {
    0: "Surgery",
    1: "chemotherapy",
    2: "radiation therapy",
    3: "close monitoring"
}


# ------------------------------------------------
# 1.6. Función auxiliar para construir el texto
# ------------------------------------------------
def build_input_text(condition: str, clinical_note: str) -> str:
    """
    Construye la cadena exactamente como fue entrenado el modelo
      "Condition: {condition}. Clinical note: {clinical_note}"
    """
    return f"Condition: {condition}. Clinical note: {clinical_note}"


# ==============================================================
# ENDPOINT 1: /predict_image
# --------------------------------------------------------------
# Recibe la imagen y devuelve únicamente 'tumor_prediction'.
# ==============================================================

@app.post("/predict_image")
async def predict_image(image_file: UploadFile = File(...)):
    """
    Recibe:
      - image_file: archivo .jpg/.png con MRI de tumor cerebral.
    Devuelve JSON:
      {
        "tumor_prediction": "brain_glioma"
      }
    """
    # 1. Validar formato de imagen
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=415,
            detail="Formato de archivo no soportado. Use JPG o PNG."
        )

    # 2. Leer y convertir la imagen a RGB
    try:
        contents = await image_file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer la imagen: {e}")

    # 3. Preprocesar la imagen
    try:
        img_cropped = image_transform(pil_image)
        img_tensor = img_cropped.unsqueeze(0).to(device)  # [1, 3, 224, 224]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en preprocesamiento de imagen: {e}")

    # 4. Inferencia del modelo de imagen
    try:
        with torch.no_grad():
            outputs_img = image_model(img_tensor)      # logits [1, 3]
            _, preds_img = torch.max(outputs_img, dim=1)
            idx_img = preds_img.item()
            tumor_label = idx2label_image[idx_img]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia de imagen: {e}")

    # 5. Devolver solo tumor_label en JSON
    return JSONResponse(content={"tumor_prediction": tumor_label})


# ==============================================================
# ENDPOINT 2: /predict_treatment
# --------------------------------------------------------------
# Recibe 'condition' (debe ser algo como "Brain Tumor") y la nota.
# Devuelve únicamente 'treatment_prediction'.
# ==============================================================

@app.post("/predict_treatment")
async def predict_treatment(
    condition: str = Form(...),
    clinical_note: str = Form(...)
):
    """
    Recibe:
      - condition: resultado previo de /predict_image (p.ej. "Brain Tumor").
      - clinical_note: nota clínica completa (texto) del paciente. En inglés, no más de 50 caracteres.
    Devuelve JSON:
      {
        "treatment_prediction": "chemotherapy"
      }
    """

    # 1. Validar que condition y clinical_note no estén vacíos
    if not condition.strip():
        raise HTTPException(status_code=400, detail="El campo 'condition' no puede estar vacío.")
    if not clinical_note.strip():
        raise HTTPException(status_code=400, detail="El campo 'clinical_note' no puede estar vacío.")

    # 2. Construir el texto de entrada para ClinicalBERT
    input_text = build_input_text(condition, clinical_note)

    # 3. Tokenizar y preprocesar el texto
    try:
        encoding = text_tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=50,           
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)           # [1, 50]
        attention_mask = encoding["attention_mask"].to(device) # [1, 50]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en tokenización de texto: {e}")

    # 4. Inferencia del modelo de texto
    try:
        with torch.no_grad():
            outputs_txt = text_model(input_ids=input_ids, attention_mask=attention_mask)
            logits_txt = outputs_txt.logits                      # [1, n_classes_text]
            _, preds_txt = torch.max(logits_txt, dim=1)
            idx_txt = preds_txt.item()

            # Mapeo seguro de idx_txt a etiqueta
            first_key = next(iter(idx2label_text.keys()))
            if isinstance(first_key, str):
                treatment_label = idx2label_text[str(idx_txt)]
            else:
                treatment_label = idx2label_text[idx_txt]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia de texto: {e}")

    # 5. Devolver solo treatment_label en JSON
    return JSONResponse(content={"treatment_prediction": treatment_label})


