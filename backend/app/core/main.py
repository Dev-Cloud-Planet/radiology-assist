# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .core.inference import analyze_xray

app = FastAPI(
    title="Radiology Assist API",
    description="API de apoyo al médico para análisis de radiografías (no diagnóstico definitivo).",
    version="0.1.0",
)

# Permitir llamadas desde cualquier origen (para pruebas locales)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_radiograph(file: UploadFile = File(...)):
    # Validación simple del tipo de archivo
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten imágenes PNG o JPG/JPEG por ahora.",
        )

    image_bytes = await file.read()

    # Llamamos a la función de inferencia (mock por ahora)
    result = analyze_xray(image_bytes)

    # Aquí dejamos muy claro que es apoyo, no diagnóstico
    result["disclaimer"] = (
        "Este resultado es generado por un sistema de IA solo como apoyo al médico. "
        "No constituye un diagnóstico definitivo ni reemplaza la valoración médica."
    )

    return result
