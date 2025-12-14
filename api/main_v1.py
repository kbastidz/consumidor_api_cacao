"""
API para Detección de Deficiencias Nutricionales en Hojas de Cacao
Endpoint: POST /predict
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

# ========================================
# CONFIGURACIÓN
# ========================================

MODEL_PATH = r'C:\Users\User\Desktop\local\modelo_final_cacao.h5'
IMG_SIZE = 224
CATEGORIAS = ['Potasio', 'Nitrogeno', 'Fosforo']

# ========================================
# CARGAR MODELO
# ========================================

print("🔄 Cargando modelo...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ No se encuentra el modelo en: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)
print(f"✅ Modelo cargado exitosamente desde: {MODEL_PATH}")

# ========================================
# CREAR APP FASTAPI
# ========================================

app = FastAPI(
    title="API Detección Deficiencias en Cacao",
    description="API para detectar deficiencias de Potasio, Nitrógeno y Fósforo en hojas de cacao",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# FUNCIÓN DE PREDICCIÓN
# ========================================

def predecir_imagen(imagen_bytes):
    """
    Procesa la imagen y retorna la predicción
    
    Args:
        imagen_bytes: Bytes de la imagen
        
    Returns:
        dict con predicción y probabilidades
    """
    try:
        # Cargar imagen desde bytes
        img = Image.open(io.BytesIO(imagen_bytes))
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convertir a array y normalizar
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predecir
        predicciones = model.predict(img_array, verbose=0)
        
        # Obtener clase predicha y confianza
        clase_idx = np.argmax(predicciones[0])
        confianza = float(predicciones[0][clase_idx] * 100)
        
        # Crear diccionario con probabilidades
        probabilidades = {
            CATEGORIAS[i]: float(predicciones[0][i] * 100)
            for i in range(len(CATEGORIAS))
        }
        
        return {
            "deficiencia": CATEGORIAS[clase_idx],
            "confianza": round(confianza, 2),
            "probabilidades": {k: round(v, 2) for k, v in probabilidades.items()}
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

# ========================================
# ENDPOINTS
# ========================================

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "mensaje": "API de Detección de Deficiencias en Cacao",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Enviar imagen para predicción",
            "GET /health": "Verificar estado del servicio",
            "GET /categorias": "Obtener categorías disponibles"
        }
    }

@app.get("/health")
async def health():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "modelo_cargado": True,
        "categorias": CATEGORIAS
    }

@app.get("/categorias")
async def obtener_categorias():
    """Obtener lista de categorías"""
    return {
        "categorias": CATEGORIAS,
        "total": len(CATEGORIAS)
    }

@app.post("/predict")
async def predecir(file: UploadFile = File(...)):
    """
    Endpoint principal para predecir deficiencias
    
    Args:
        file: Imagen de la hoja de cacao (JPG, JPEG, PNG)
        
    Returns:
        JSON con la predicción y probabilidades
    """
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (JPG, JPEG, PNG)"
        )
    
    try:
        # Leer bytes de la imagen
        imagen_bytes = await file.read()
        
        # Realizar predicción
        resultado = predecir_imagen(imagen_bytes)
        
        return JSONResponse(content={
            "success": True,
            "data": resultado,
            "archivo": file.filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# EJECUTAR
# ========================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Iniciando API de Detección de Deficiencias en Cacao")
    print("="*60)
    print(f"📍 URL: http://localhost:8000")
    print(f"📖 Documentación: http://localhost:8000/docs")
    print(f"📊 Categorías: {', '.join(CATEGORIAS)}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)