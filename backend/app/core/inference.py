# backend/app/core/inference.py

import io
from typing import List, Dict

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torchxrayvision as xrv
from skimage import img_as_float  # por si quieres usarlo luego
from torchxrayvision import datasets as xrv_datasets

# ---------- CONFIG GLOBAL DEL MODELO ----------

# En Docker estamos usando PyTorch CPU
_DEVICE = torch.device("cpu")
_MODEL = None  # se cargará perezosamente la primera vez

# Cuántas patologías como máximo mostramos en el informe
TOP_K_PATHOLOGIES = 3

# Umbrales por patología (probabilidad mínima para aparecer en el informe)
# Estos valores son un punto de partida, luego los podréis ajustar con datos reales.
PATHOLOGY_THRESHOLDS = {
    # Hallazgos críticos → umbral más alto
    "Pneumothorax": 0.80,
    "Fracture": 0.80,

    # Hallazgos importantes
    "Effusion": 0.75,               # derrame pleural
    "Pneumonia": 0.75,
    "Edema": 0.75,                  # edema pulmonar
    "Lung Lesion": 0.75,
    "Lung Opacity": 0.75,
    "Hernia": 0.75,

    # Hallazgos frecuentes, algo menos estrictos
    "Cardiomegaly": 0.70,
    "Mass": 0.70,
    "Nodule": 0.70,
    "Atelectasis": 0.70,
    "Consolidation": 0.70,
    "Infiltration": 0.70,
    "Pleural_Thickening": 0.70,
    "Emphysema": 0.70,
    "Fibrosis": 0.70,
    "Enlarged Cardiomediastinum": 0.70,

    # Umbral por defecto si una patología no está en el diccionario
    "_default": 0.75,
}

# Traducciones EN -> ES para el informe
PATHOLOGY_TRANSLATIONS = {
    "Atelectasis": "atelectasia",
    "Consolidation": "consolidación",
    "Infiltration": "infiltrados",
    "Pneumothorax": "neumotórax",
    "Edema": "edema pulmonar",
    "Emphysema": "enfisema",
    "Fibrosis": "fibrosis pulmonar",
    "Effusion": "derrame pleural",
    "Pneumonia": "neumonía",
    "Pleural_Thickening": "engrosamiento pleural",
    "Cardiomegaly": "cardiomegalia",
    "Nodule": "nódulo pulmonar",
    "Mass": "masa pulmonar",
    "Hernia": "hernia",
    "Lung Lesion": "lesión pulmonar",
    "Lung Opacity": "opacidad pulmonar",
    "Enlarged Cardiomediastinum": "aumento del mediastino",
    "Fracture": "fractura ósea",
}


def _load_model():
    """
    Carga el modelo pre-entrenado de TorchXRayVision una sola vez.
    Usamos DenseNet-121 entrenado en varios datasets de tórax.
    """
    global _MODEL
    if _MODEL is None:
        # Este peso incluye múltiples datasets ("densenet121-res224-all")
        _MODEL = xrv.models.DenseNet(weights="densenet121-res224-all")
        _MODEL.eval()
        _MODEL.to(_DEVICE)
    return _MODEL


def _preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocesa la radiografía siguiendo el pipeline recomendado por TorchXRayVision:
    - Leer imagen
    - Normalizar a rango [-1024, 1024]
    - Escala de grises
    - CenterCrop + Resize
    - Convertir a tensor [1,1,H,W]
    """
    # Cargar con PIL y convertir a escala de grises
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # A numpy float32
    img = np.array(pil_img).astype(np.float32)

    # Normalizar como indica la doc: xrv_datasets.normalize(img, 255)
    img = xrv_datasets.normalize(img, 255)  # 8-bit -> [-1024, 1024]

    # Añadimos canal: [1, H, W]
    if img.ndim == 2:
        img = img[None, ...]  # (1, H, W)

    # Transformaciones de recorte y resize recomendadas
    transform = T.Compose(
        [
            xrv_datasets.XRayCenterCrop(),
            xrv_datasets.XRayResizer(224),
        ]
    )
    img = transform(img)  # sigue siendo numpy

    # A tensor [1,1,H,W]
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # batch=1
    img_tensor = img_tensor.to(_DEVICE).float()

    return img_tensor


def _get_threshold_for_label(label_en: str) -> float:
    """
    Devuelve el umbral específico para una patología, o el _default si no está definido.
    """
    return PATHOLOGY_THRESHOLDS.get(label_en, PATHOLOGY_THRESHOLDS["_default"])


def _generate_spanish_report(sorted_preds: List[Dict]) -> str:
    """
    Genera un informe preliminar en español a partir de las predicciones.
    Usa umbrales específicos por patología y muestra como máximo TOP_K_PATHOLOGIES.
    """
    lines = []
    lines.append("Informe automático de apoyo al médico (NO definitivo):\n")

    # Filtramos cada patología según su umbral individual
    candidates = []
    for p in sorted_preds:
        label_en = p["label_en"]
        prob = p["probability"]
        threshold = _get_threshold_for_label(label_en)
        if prob >= threshold:
            candidates.append(p)

    # Nos quedamos solo con las TOP_K_PATHOLOGIES más probables
    top_preds = candidates[:TOP_K_PATHOLOGIES]

    if top_preds:
        lines.append("Hallazgos principales sugeridos por el modelo:")
        for p in top_preds:
            label_es = p["label_es"]
            prob_pct = p["probability"] * 100
            lines.append(f"- {label_es} (probabilidad estimada: {prob_pct:.1f}%)")
    else:
        lines.append(
            "No se identifican hallazgos patológicos relevantes con alta "
            "probabilidad según el modelo. Esto NO excluye enfermedad."
        )

    lines.append("")
    lines.append(
        "Este informe está generado por un sistema de IA entrenado en radiografías "
        "de tórax. Debe interpretarse siempre junto con la clínica y la valoración "
        "del médico responsable."
    )

    return "\n".join(lines)


def analyze_xray(image_bytes: bytes) -> dict:
    """
    Recibe una radiografía de tórax (PNG/JPG) en bytes y devuelve:
    - lista de patologías con probabilidades (predictions)
    - informe preliminar en español (preliminary_report)
    - info del modelo (model_info)
    """
    model = _load_model()
    img_tensor = _preprocess_image(image_bytes)

    with torch.no_grad():
        outputs = model(img_tensor)  # [1, num_pathologies]

    # TorchXRayVision devuelve scores en rango [0,1] para cada patología
    probs = outputs[0].detach().cpu().numpy().tolist()

    # Obtener nombres de patologías
    pathologies = getattr(model, "pathologies", None)
    if pathologies is None:
        pathologies = getattr(model, "targets", [])

    preds = []
    for label_en, prob in zip(pathologies, probs):
        label_es = PATHOLOGY_TRANSLATIONS.get(label_en, label_en)
        preds.append(
            {
                "label_en": label_en,
                "label_es": label_es,
                "probability": float(prob),
            }
        )

    # Ordenamos de mayor a menor probabilidad
    preds_sorted = sorted(preds, key=lambda x: x["probability"], reverse=True)

    # Generar informe corto usando umbrales por patología
    report = _generate_spanish_report(preds_sorted)

    return {
        "predictions": preds_sorted,
        "preliminary_report": report,
        "model_info": {
            "library": "torchxrayvision",
            "architecture": "DenseNet-121",
            "weights": "densenet121-res224-all",
            "device": str(_DEVICE),
        },
    }
