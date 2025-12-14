"""
API para Detección de Deficiencias Nutricionales en Hojas de Cacao
CON VALIDACIÓN DE IMÁGENES NO-CACAO
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

# ⚠️ IMPORTANTE: Actualiza esta ruta al nuevo modelo con validación
MODEL_PATH = r'C:\Users\User\Desktop\local\modelo_final_cacao_validado.h5'
IMG_SIZE = 224

# Nueva configuración con 4 categorías (incluye No-Cacao)
CATEGORIAS = ['Potasio', 'Nitrogeno', 'Fosforo', 'No-Cacao']

# Umbrales de validación
CONFIANZA_MINIMA = 0.60  # 60% - Ajustable según necesidades
DIFERENCIA_MINIMA_PROBABILIDADES = 0.15  # 15% diferencia entre top 2 clases

# ========================================
# CARGAR MODELO
# ========================================

print("🔄 Cargando modelo con validación...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ No se encuentra el modelo en: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)
print(f"✅ Modelo cargado exitosamente desde: {MODEL_PATH}")
print(f"📊 Categorías: {CATEGORIAS}")
print(f"⚙️ Umbral de confianza: {CONFIANZA_MINIMA * 100}%")

# ========================================
# CREAR APP FASTAPI
# ========================================

app = FastAPI(
    title="API Detección Deficiencias en Cacao (con Validación)",
    description="API para detectar deficiencias de Potasio, Nitrógeno y Fósforo en hojas de cacao. Incluye validación para rechazar imágenes que no sean hojas de cacao.",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# FUNCIÓN DE PREDICCIÓN CON VALIDACIÓN
# ========================================

def predecir_imagen_validada(imagen_bytes):
    """
    Procesa la imagen y retorna la predicción CON VALIDACIÓN
    
    Args:
        imagen_bytes: Bytes de la imagen
        
    Returns:
        dict con predicción, validación y probabilidades
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
        confianza = float(predicciones[0][clase_idx])
        categoria_predicha = CATEGORIAS[clase_idx]
        
        # Crear diccionario con probabilidades
        probabilidades = {
            CATEGORIAS[i]: float(predicciones[0][i] * 100)
            for i in range(len(CATEGORIAS))
        }
        
        # ========================================
        # SISTEMA DE VALIDACIÓN MULTI-NIVEL
        # ========================================
        
        es_valido = True
        mensaje_validacion = "Imagen válida - Hoja de cacao detectada"
        tipo_alerta = "success"
        nivel_confianza = "alta"
        
        # NIVEL 1: Verificar si es clasificado como No-Cacao
        if categoria_predicha == 'No-Cacao':
            es_valido = False
            mensaje_validacion = "Esta imagen NO parece ser una hoja de cacao"
            tipo_alerta = "no-cacao"
            nivel_confianza = "invalido"
        
        # NIVEL 2: Verificar umbral de confianza
        elif confianza < CONFIANZA_MINIMA:
            es_valido = False
            mensaje_validacion = f"Confianza baja ({confianza*100:.1f}%). Imagen ambigua o no es hoja de cacao"
            tipo_alerta = "baja-confianza"
            nivel_confianza = "baja"
        
        # NIVEL 3: Verificar distribución de probabilidades
        prob_ordenadas = np.sort(predicciones[0])[::-1]
        diferencia_top2 = prob_ordenadas[0] - prob_ordenadas[1]
        
        if diferencia_top2 < DIFERENCIA_MINIMA_PROBABILIDADES:
            if es_valido:
                # No invalida, pero advierte
                mensaje_validacion = f"Predicción: {categoria_predicha}. Advertencia: probabilidades dispersas, imagen poco clara"
                tipo_alerta = "warning"
                nivel_confianza = "media"
            else:
                mensaje_validacion += " | Probabilidades muy dispersas"
        
        # ========================================
        # CONSTRUIR RESPUESTA
        # ========================================
        
        resultado = {
            "es_valido": es_valido,
            "deficiencia": categoria_predicha if es_valido else None,
            "confianza": round(confianza * 100, 2),
            "nivel_confianza": nivel_confianza,
            "mensaje": mensaje_validacion,
            "tipo_alerta": tipo_alerta,
            "probabilidades": {k: round(v, 2) for k, v in probabilidades.items()},
            "metadata": {
                "diferencia_top2": round(diferencia_top2 * 100, 2),
                "umbral_confianza": CONFIANZA_MINIMA * 100,
                "categoria_predicha_raw": categoria_predicha  # Siempre muestra qué detectó
            }
        }
        
        # Si no es válido, agregar recomendaciones
        if not es_valido:
            resultado["recomendaciones"] = [
                "Asegúrate de que la imagen contenga una hoja de cacao",
                "Verifica que la imagen esté bien iluminada y enfocada",
                "Evita fondos con muchos elementos distractores",
                "Toma la foto desde arriba de la hoja completa"
            ]
        
        return resultado
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

# ========================================
# ENDPOINTS
# ========================================

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "mensaje": "API de Detección de Deficiencias en Cacao con Validación",
        "version": "2.0.0",
        "caracteristicas": [
            "Detección de deficiencias: Potasio, Nitrógeno, Fósforo",
            "Validación automática de imágenes no-cacao",
            "Sistema de confianza multi-nivel",
            "Rechazo de imágenes ambiguas"
        ],
        "endpoints": {
            "POST /predict": "Enviar imagen para predicción con validación",
            "GET /health": "Verificar estado del servicio",
            "GET /categorias": "Obtener categorías disponibles",
            "GET /configuracion": "Ver configuración de umbrales"
        }
    }

@app.get("/health")
async def health():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "modelo_cargado": True,
        "modelo_version": "2.0.0 (con validación)",
        "categorias": CATEGORIAS,
        "validacion_activa": True
    }

@app.get("/categorias")
async def obtener_categorias():
    """Obtener lista de categorías"""
    return {
        "categorias": CATEGORIAS,
        "categorias_deficiencias": [c for c in CATEGORIAS if c != 'No-Cacao'],
        "total": len(CATEGORIAS),
        "incluye_validacion": True
    }

@app.get("/configuracion")
async def obtener_configuracion():
    """Obtener configuración de umbrales"""
    return {
        "umbral_confianza_minima": CONFIANZA_MINIMA * 100,
        "diferencia_minima_probabilidades": DIFERENCIA_MINIMA_PROBABILIDADES * 100,
        "imagen_size": IMG_SIZE,
        "descripcion": {
            "umbral_confianza": "Confianza mínima requerida para considerar válida una predicción",
            "diferencia_probabilidades": "Diferencia mínima entre las dos probabilidades más altas"
        }
    }

@app.post("/predict")
async def predecir(file: UploadFile = File(...)):
    """
    Endpoint principal para predecir deficiencias CON VALIDACIÓN
    
    Args:
        file: Imagen de la hoja de cacao (JPG, JPEG, PNG)
        
    Returns:
        JSON con la predicción, validación y probabilidades
        
    Ejemplo de respuesta exitosa (válida):
    {
        "success": true,
        "data": {
            "es_valido": true,
            "deficiencia": "Potasio",
            "confianza": 85.5,
            "nivel_confianza": "alta",
            "mensaje": "Imagen válida - Hoja de cacao detectada",
            "tipo_alerta": "success",
            "probabilidades": {...}
        }
    }
    
    Ejemplo de respuesta (no válida):
    {
        "success": true,
        "data": {
            "es_valido": false,
            "deficiencia": null,
            "confianza": 73.42,
            "mensaje": "Esta imagen NO parece ser una hoja de cacao",
            "tipo_alerta": "no-cacao",
            "recomendaciones": [...]
        }
    }
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
        
        # Realizar predicción con validación
        resultado = predecir_imagen_validada(imagen_bytes)
        
        return JSONResponse(content={
            "success": True,
            "data": resultado,
            "archivo": file.filename,
            "timestamp": str(np.datetime64('now'))
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predecir_batch(files: list[UploadFile] = File(...)):
    """
    Endpoint para predecir múltiples imágenes a la vez
    
    Args:
        files: Lista de imágenes
        
    Returns:
        JSON con resultados de todas las predicciones
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Máximo 10 imágenes por solicitud"
        )
    
    resultados = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            resultados.append({
                "archivo": file.filename,
                "error": "No es una imagen válida"
            })
            continue
        
        try:
            imagen_bytes = await file.read()
            resultado = predecir_imagen_validada(imagen_bytes)
            resultados.append({
                "archivo": file.filename,
                "resultado": resultado
            })
        except Exception as e:
            resultados.append({
                "archivo": file.filename,
                "error": str(e)
            })
    
    # Estadísticas del batch
    total = len(resultados)
    validas = sum(1 for r in resultados if r.get('resultado', {}).get('es_valido', False))
    invalidas = total - validas
    
    return JSONResponse(content={
        "success": True,
        "total_imagenes": total,
        "imagenes_validas": validas,
        "imagenes_invalidas": invalidas,
        "resultados": resultados
    })

# ========================================
# EJECUTAR
# ========================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 API de Detección de Deficiencias en Cacao v2.0")
    print("   CON SISTEMA DE VALIDACIÓN")
    print("="*60)
    print(f"📍 URL: http://localhost:8000")
    print(f"📖 Documentación: http://localhost:8000/docs")
    print(f"📊 Categorías: {', '.join(CATEGORIAS)}")
    print(f"✅ Validación: Activa")
    print(f"⚙️  Umbral confianza: {CONFIANZA_MINIMA * 100}%")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)