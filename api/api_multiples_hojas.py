"""
API para Detección de Deficiencias Nutricionales en Hojas de Cacao
USANDO YOLO - Detección de Objetos
Endpoint: POST /predict
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64
from ultralytics import YOLO
from datetime import datetime

# ========================================
# CONFIGURACIÓN
# ========================================

#YOLO_MODEL_PATH = r'runs/detect/cacao_detector/weights/best.pt'
YOLO_MODEL_PATH = r"C:\Users\User\Documents\GitHub\consumidor_api_cacao\local\runs\detect\cacao_detector\weights\best.pt"
OUTPUT_DIR = "resultados_api"

# Umbrales de detección
CONF_THRESHOLD = 0.15   # Confianza mínima para detectar
IOU_THRESHOLD = 0.60    # Umbral de IoU para NMS

# Umbral para considerar válida la detección
CONFIANZA_MINIMA_VALIDA = 0.25  # 25% - ajustable

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# CARGAR MODELO YOLO
# ========================================

print("🔄 Cargando modelo YOLO...")
if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"❌ No se encuentra el modelo en: {YOLO_MODEL_PATH}")

modelo = YOLO(YOLO_MODEL_PATH)
CLASES = modelo.names

print(f"✅ Modelo YOLO cargado exitosamente")
print(f"📊 Clases detectables: {CLASES}")
print(f"⚙️ Umbral de confianza: {CONF_THRESHOLD}")

# ========================================
# CREAR APP FASTAPI
# ========================================

app = FastAPI(
    title="API Detección de Deficiencias en Cacao - YOLO",
    description="API para detectar deficiencias nutricionales en hojas de cacao usando YOLOv8",
    version="3.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# FUNCIONES AUXILIARES
# ========================================

def imagen_a_base64(img_array):
    """Convierte imagen numpy a base64"""
    is_success, buffer = cv2.imencode(".jpg", img_array)
    if not is_success:
        return None
    return base64.b64encode(buffer).decode('utf-8')

def dibujar_detecciones(img, boxes):
    """Dibuja las cajas de detección en la imagen"""
    img_draw = img.copy()
    
    if boxes is None or len(boxes) == 0:
        return img_draw
    
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    clases = boxes.cls.cpu().numpy().astype(int)
    
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        clase_id = clases[i]
        nombre = CLASES[clase_id]
        confianza = confs[i] * 100
        
        # Dibujar rectángulo
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Crear etiqueta
        label = f"{nombre} {confianza:.1f}%"
        
        # Fondo para el texto
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_draw, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        
        # Texto
        cv2.putText(img_draw, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img_draw

# ========================================
# FUNCIÓN DE PREDICCIÓN PRINCIPAL
# ========================================

def predecir_deficiencias_yolo(imagen_bytes, incluir_imagen=False):
    """
    Procesa la imagen y retorna las detecciones
    
    Args:
        imagen_bytes: Bytes de la imagen
        incluir_imagen: Si True, incluye la imagen con detecciones en base64
        
    Returns:
        dict con detecciones y metadatos
    """
    try:
        # Cargar imagen desde bytes
        img_pil = Image.open(io.BytesIO(imagen_bytes))
        
        # Convertir a RGB si es necesario
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        # Convertir a formato OpenCV (numpy array)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # ========================================
        # INFERENCIA YOLO
        # ========================================
        
        results = modelo(
            img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        
        boxes = results[0].boxes
        
        # ========================================
        # PROCESAR RESULTADOS
        # ========================================
        
        detecciones = []
        es_valido = True
        mensaje = ""
        tipo_alerta = "success"
        
        if boxes is None or len(boxes) == 0:
            es_valido = False
            mensaje = "No se detectaron deficiencias en la hoja de cacao"
            tipo_alerta = "no-deteccion"
        else:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            clases = boxes.cls.cpu().numpy().astype(int)
            
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                clase_id = int(clases[i])
                nombre = CLASES[clase_id]
                confianza = float(confs[i])
                
                detecciones.append({
                    "deficiencia": nombre,
                    "confianza": round(confianza * 100, 2),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "ancho": int(x2 - x1),
                        "alto": int(y2 - y1)
                    },
                    "area": int((x2 - x1) * (y2 - y1))
                })
            
            # Determinar mensaje y validación
            max_confianza = max([d["confianza"] for d in detecciones])
            
            if max_confianza < CONFIANZA_MINIMA_VALIDA * 100:
                tipo_alerta = "warning"
                mensaje = f"Detecciones con baja confianza (máx: {max_confianza:.1f}%). Verifica la calidad de la imagen"
            else:
                mensaje = f"Se detectaron {len(detecciones)} deficiencia(s) en la hoja"
        
        # ========================================
        # ESTADÍSTICAS
        # ========================================
        
        estadisticas = {
            "total_detecciones": len(detecciones),
            "deficiencias_unicas": len(set([d["deficiencia"] for d in detecciones])),
            "confianza_promedio": round(np.mean([d["confianza"] for d in detecciones]), 2) if detecciones else 0,
            "confianza_maxima": round(max([d["confianza"] for d in detecciones]), 2) if detecciones else 0
        }
        
        # Contar por tipo de deficiencia
        conteo_deficiencias = {}
        for det in detecciones:
            nombre = det["deficiencia"]
            conteo_deficiencias[nombre] = conteo_deficiencias.get(nombre, 0) + 1
        
        estadisticas["por_tipo"] = conteo_deficiencias
        
        # ========================================
        # CONSTRUIR RESPUESTA
        # ========================================
        
        resultado = {
            "es_valido": es_valido,
            "mensaje": mensaje,
            "tipo_alerta": tipo_alerta,
            "detecciones": detecciones,
            "estadisticas": estadisticas,
            "metadata": {
                "dimensiones_imagen": {
                    "ancho": img.shape[1],
                    "alto": img.shape[0]
                },
                "umbral_confianza": CONF_THRESHOLD,
                "umbral_iou": IOU_THRESHOLD
            }
        }
        
        # Incluir imagen con detecciones si se solicita
        if incluir_imagen and boxes is not None and len(boxes) > 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_con_detecciones = dibujar_detecciones(img_rgb, boxes)
            img_con_detecciones_bgr = cv2.cvtColor(img_con_detecciones, cv2.COLOR_RGB2BGR)
            resultado["imagen_procesada"] = imagen_a_base64(img_con_detecciones_bgr)
        
        # Recomendaciones si no hay detecciones válidas
        if not es_valido or tipo_alerta == "warning":
            resultado["recomendaciones"] = [
                "Asegúrate de que la imagen muestre claramente una hoja de cacao",
                "Verifica que la iluminación sea adecuada",
                "Evita sombras y reflejos excesivos",
                "Captura la hoja completa en el encuadre",
                "Mantén la cámara estable para evitar desenfoques"
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
        "mensaje": "API de Detección de Deficiencias en Cacao - YOLO",
        "version": "3.0.0",
        "modelo": "YOLOv8",
        "caracteristicas": [
            "Detección de múltiples deficiencias en una misma hoja",
            "Localización precisa (bounding boxes)",
            "Análisis de múltiples regiones",
            "Estadísticas por tipo de deficiencia"
        ],
        "clases_detectables": list(CLASES.values()),
        "endpoints": {
            "POST /predict": "Enviar imagen para detección de deficiencias",
            "POST /predict/visual": "Igual que /predict pero incluye imagen procesada",
            "GET /health": "Verificar estado del servicio",
            "GET /clases": "Obtener clases detectables",
            "GET /configuracion": "Ver configuración de umbrales"
        }
    }

@app.get("/health")
async def health():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "modelo_cargado": True,
        "modelo_tipo": "YOLOv8",
        "version": "3.0.0",
        "clases": list(CLASES.values())
    }

@app.get("/clases")
async def obtener_clases():
    """Obtener lista de clases detectables"""
    return {
        "clases": CLASES,
        "total": len(CLASES),
        "descripcion": "Deficiencias nutricionales detectables en hojas de cacao"
    }

@app.get("/configuracion")
async def obtener_configuracion():
    """Obtener configuración de umbrales"""
    return {
        "umbral_confianza": CONF_THRESHOLD,
        "umbral_iou": IOU_THRESHOLD,
        "confianza_minima_valida": CONFIANZA_MINIMA_VALIDA,
        "descripcion": {
            "umbral_confianza": "Confianza mínima para que YOLO detecte un objeto",
            "umbral_iou": "Umbral de Intersection over Union para Non-Maximum Suppression",
            "confianza_minima_valida": "Confianza mínima para considerar una detección como válida"
        }
    }

@app.post("/predict")
async def predecir(file: UploadFile = File(...)):
    """
    Endpoint principal para detectar deficiencias en hojas de cacao
    
    Args:
        file: Imagen de la hoja de cacao (JPG, JPEG, PNG)
        
    Returns:
        JSON con las detecciones, estadísticas y metadatos
        
    Ejemplo de respuesta exitosa:
    {
        "success": true,
        "data": {
            "es_valido": true,
            "mensaje": "Se detectaron 2 deficiencia(s) en la hoja",
            "tipo_alerta": "success",
            "detecciones": [
                {
                    "deficiencia": "Potasio",
                    "confianza": 87.5,
                    "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 350},
                    "area": 40000
                }
            ],
            "estadisticas": {
                "total_detecciones": 2,
                "deficiencias_unicas": 2,
                "confianza_promedio": 85.3,
                "por_tipo": {"Potasio": 1, "Nitrogeno": 1}
            }
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
        
        # Realizar detección
        resultado = predecir_deficiencias_yolo(imagen_bytes, incluir_imagen=False)
        
        return JSONResponse(content={
            "success": True,
            "data": resultado,
            "archivo": file.filename,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/visual")
async def predecir_con_visual(file: UploadFile = File(...)):
    """
    Endpoint para detectar deficiencias E INCLUIR imagen procesada con cajas
    
    Returns:
        JSON con detecciones + imagen procesada en base64
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (JPG, JPEG, PNG)"
        )
    
    try:
        imagen_bytes = await file.read()
        resultado = predecir_deficiencias_yolo(imagen_bytes, incluir_imagen=True)
        
        return JSONResponse(content={
            "success": True,
            "data": resultado,
            "archivo": file.filename,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predecir_batch(files: list[UploadFile] = File(...)):
    """
    Endpoint para detectar deficiencias en múltiples imágenes
    
    Args:
        files: Lista de imágenes (máximo 10)
        
    Returns:
        JSON con resultados de todas las detecciones
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
            resultado = predecir_deficiencias_yolo(imagen_bytes, incluir_imagen=False)
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
    con_detecciones = sum(1 for r in resultados 
                          if 'resultado' in r and r['resultado'].get('es_valido', False))
    sin_detecciones = total - con_detecciones
    
    total_deficiencias = sum(
        len(r['resultado']['detecciones']) 
        for r in resultados 
        if 'resultado' in r
    )
    
    return JSONResponse(content={
        "success": True,
        "total_imagenes": total,
        "imagenes_con_detecciones": con_detecciones,
        "imagenes_sin_detecciones": sin_detecciones,
        "total_deficiencias_detectadas": total_deficiencias,
        "resultados": resultados
    })

# ========================================
# EJECUTAR
# ========================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 API de Detección de Deficiencias en Cacao v3.0")
    print("   MODELO: YOLOv8 - Detección de Objetos")
    print("="*60)
    print(f"📍 URL: http://localhost:8000")
    print(f"📖 Documentación: http://localhost:8000/docs")
    print(f"📊 Clases: {', '.join(CLASES.values())}")
    print(f"⚙️  Umbral confianza: {CONF_THRESHOLD}")
    print(f"⚙️  Umbral IoU: {IOU_THRESHOLD}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)