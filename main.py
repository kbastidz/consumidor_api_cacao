# ========================================
# API PARA DETECCIÓN DE DEFICIENCIAS EN CACAO
# Deploy en Render
# ========================================

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from typing import Dict

app = FastAPI(title="Cacao Nutrition Detection API")

# Configurar CORS para permitir peticiones desde tu web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
IMG_SIZE = 224
CATEGORIAS = ['Potasio', 'Nitrogeno', 'Fosforo']

# Cargar modelo al iniciar la API
print("🔄 Cargando modelo...")
modelo = load_model('modelo_final_cacao.h5')
print("✅ Modelo cargado exitosamente")

@app.get("/")
async def root():
    """Endpoint raíz para verificar que la API está funcionando"""
    return {
        "status": "online",
        "message": "API de Detección de Deficiencias en Cacao",
        "version": "1.0",
        "categorias": CATEGORIAS
    }

@app.get("/health")
async def health_check():
    """Health check para Render"""
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Endpoint principal para predecir deficiencias
    
    Args:
        file: Imagen de hoja de cacao (JPG, PNG)
    
    Returns:
        JSON con predicciones y probabilidades
    """
    try:
        # Leer imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocesar
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predecir
        predicciones = modelo.predict(img_array, verbose=0)
        clase_predicha = int(np.argmax(predicciones[0]))
        confianza = float(predicciones[0][clase_predicha])
        
        # Formatear respuesta
        response = {
            "success": True,
            "prediccion": {
                "deficiencia": CATEGORIAS[clase_predicha],
                "confianza": round(confianza * 100, 2),
                "confianza_decimal": round(confianza, 4)
            },
            "probabilidades": {
                cat: round(float(prob) * 100, 2) 
                for cat, prob in zip(CATEGORIAS, predicciones[0])
            },
            "detalles": {
                "clase_index": clase_predicha,
                "imagen_procesada": f"{IMG_SIZE}x{IMG_SIZE}",
                "formato_original": file.content_type
            }
        }
        
        return response
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Error al procesar la imagen"
        }

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predecir múltiples imágenes a la vez"""
    resultados = []
    
    for file in files:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predicciones = modelo.predict(img_array, verbose=0)
            clase_predicha = int(np.argmax(predicciones[0]))
            
            resultados.append({
                "archivo": file.filename,
                "deficiencia": CATEGORIAS[clase_predicha],
                "confianza": round(float(predicciones[0][clase_predicha]) * 100, 2)
            })
        except Exception as e:
            resultados.append({
                "archivo": file.filename,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_imagenes": len(files),
        "resultados": resultados
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)