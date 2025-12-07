from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .core.inference import analyze_xray

# 1. IMPORTAR LA CLASE NECESARIA
from fastapi.staticfiles import StaticFiles 

app = FastAPI(
    title="Radiology Assist API",
    description=(
        "API de apoyo al médico para análisis de radiografías de tórax "
        "(no diagnóstico definitivo)."
    ),
    version="0.2.0",
)

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
    # Por ahora, solo aceptamos PNG/JPEG exportadas del sistema de la clínica.
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten imágenes PNG o JPG/JPEG por ahora.",
        )

    image_bytes = await file.read()

    try:
        result = analyze_xray(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la radiografía: {str(e)}",
        )

    # Dejamos MUY claro el rol de la IA
    result["disclaimer"] = (
        "Este resultado es generado por un sistema de IA entrenado en "
        "radiografías de tórax y se proporciona únicamente como apoyo al médico. "
        "NO constituye un diagnóstico definitivo ni reemplaza la valoración clínica."
    )

    return result

# 2. MONTAR ARCHIVOS ESTÁTICOS AL FINAL (¡CRÍTICO!)
# Esto debe ir siempre al final de todas las rutas de la API.
# directory="." apunta a la raíz del proyecto (donde está index.html).
# html=True le indica a FastAPI que sirva index.html cuando se accede a '/'.
app.mount(
    "/",
    StaticFiles(directory=".", html=True),
    name="static"
)