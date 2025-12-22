# ========================================
# DETECTOR DE MÚLTIPLES HOJAS DE CACAO CON YOLO
# Detecta múltiples hojas en una imagen y clasifica cada una
# FIX: Compatible con PyTorch 2.6+
# ========================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURACIÓN
# ========================================

BASE_PATH = r'C:\Users\User\Documents\GitHub\consumidor_api_cacao\local'
MODELO_CLASIFICADOR = os.path.join(BASE_PATH, 'modelo_final_cacao_validado.h5')

# Parámetros del clasificador
IMG_SIZE = 224
CATEGORIAS = ['Potasio', 'Nitrogeno', 'Fosforo', 'No-Cacao']
CONFIANZA_MINIMA = 0.40
#0.60

# Parámetros YOLO
YOLO_CONF_THRESHOLD = 0.25  # Confianza mínima para detección
YOLO_IOU_THRESHOLD = 0.45   # Umbral para NMS (Non-Maximum Suppression)

print("="*70)
print("🌿 DETECTOR DE MÚLTIPLES HOJAS DE CACAO")
print("   Usando YOLOv8 + Clasificador de Deficiencias")
print("="*70)

# ========================================
# VERIFICAR DEPENDENCIAS
# ========================================

try:
    from ultralytics import YOLO
    print("\n✅ YOLOv8 detectado correctamente")
except ImportError:
    print("\n❌ ERROR: No se encontró ultralytics")
    print("\n📦 Instala YOLOv8 con:")
    print("   pip install ultralytics")
    print("\n   Otras dependencias opcionales:")
    print("   pip install opencv-python")
    exit()

# ========================================
# FIX PARA PYTORCH 2.6+
# ========================================

try:
    import torch
    # Agregar clases de ultralytics como globales seguros
    torch.serialization.add_safe_globals([
        'ultralytics.nn.tasks.DetectionModel',
        'ultralytics.nn.modules.conv.Conv',
        'ultralytics.nn.modules.block.Bottleneck',
        'ultralytics.nn.modules.block.C2f',
        'ultralytics.nn.modules.block.SPPF',
        'ultralytics.nn.modules.block.Concat',
        'ultralytics.nn.modules.head.Detect',
    ])
    print("   ✓ PyTorch safe globals configurado")
except Exception as e:
    print(f"   ⚠️  No se pudo configurar safe_globals: {e}")
    print("   Continuando de todas formas...")

# ========================================
# CARGAR MODELOS
# ========================================

print("\n🔄 Cargando modelos...")

# 1. Cargar clasificador de deficiencias
if not os.path.exists(MODELO_CLASIFICADOR):
    print(f"❌ No se encuentra el clasificador en: {MODELO_CLASIFICADOR}")
    print("   Entrena primero el modelo de clasificación")
    exit()

clasificador = keras.models.load_model(MODELO_CLASIFICADOR)
print(f"   ✓ Clasificador cargado: {MODELO_CLASIFICADOR}")

# 2. Cargar YOLO preentrenado
# Opciones: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
print("\n   Descargando/cargando YOLOv8...")

try:
    detector_yolo = YOLO('yolov8n.pt')
    print("   ✓ YOLOv8 listo")
except Exception as e:
    print(f"\n❌ ERROR al cargar YOLO: {e}")
    print("\n💡 SOLUCIÓN:")
    print("   Instala PyTorch 2.5.1 (compatible):")
    print("   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu")
    exit()

print("\n💡 NOTA: Este YOLO está pre-entrenado en COCO dataset")
print("   Para mejor precisión, deberías entrenarlo con tus hojas de cacao")
print("   Ver instrucciones al final del script")

# ========================================
# FUNCIÓN PRINCIPAL: DETECTAR Y CLASIFICAR
# ========================================

def detectar_y_clasificar_hojas(ruta_imagen, 
                                 mostrar_resultados=True,
                                 guardar_resultados=True,
                                 filtrar_solo_plantas=True):
    """
    Detecta múltiples hojas en una imagen y clasifica cada una
    
    Args:
        ruta_imagen: Ruta a la imagen
        mostrar_resultados: Mostrar visualización
        guardar_resultados: Guardar imagen con detecciones
        filtrar_solo_plantas: Solo procesar objetos tipo planta detectados por YOLO
    
    Returns:
        dict con resultados de todas las hojas detectadas
    """
    
    if not os.path.exists(ruta_imagen):
        print(f"❌ No se encuentra: {ruta_imagen}")
        return None
    
    print("\n" + "="*70)
    print(f"📸 Procesando: {os.path.basename(ruta_imagen)}")
    print("="*70)
    
    # Cargar imagen
    img_original = Image.open(ruta_imagen)
    img_cv = cv2.imread(ruta_imagen)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # ========================================
    # PASO 1: DETECTAR OBJETOS CON YOLO
    # ========================================
    
    print("\n🔍 PASO 1: Detectando objetos con YOLO...")
    
    resultados_yolo = detector_yolo(ruta_imagen, 
                                    conf=YOLO_CONF_THRESHOLD,
                                    iou=YOLO_IOU_THRESHOLD,
                                    verbose=False)
    
    detecciones = resultados_yolo[0].boxes
    
    # Clases de COCO que podrían ser hojas/plantas (ajustar según necesites)
    # 0: person, 47: apple, 51: banana, 56: potted plant, etc.
    CLASES_PLANTAS = [56]  # potted plant - ajusta según tus necesidades
    
    if filtrar_solo_plantas:
        print(f"   Filtrando solo objetos tipo planta (clases: {CLASES_PLANTAS})")
    
    # Procesar detecciones
    hojas_detectadas = []
    
    for i, box in enumerate(detecciones):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        
        # Filtrar por tipo de objeto si está activado
        if filtrar_solo_plantas and cls not in CLASES_PLANTAS:
            continue
        
        nombre_clase = detector_yolo.names[cls]
        
        hojas_detectadas.append({
            'id': i,
            'bbox': (x1, y1, x2, y2),
            'confianza_deteccion': conf * 100,
            'clase_yolo': nombre_clase,
            'area': (x2-x1) * (y2-y1)
        })
    
    num_detecciones = len(hojas_detectadas)
    print(f"\n   ✓ Detectados {num_detecciones} objeto(s) potencial(es)")
    
    if num_detecciones == 0:
        print("\n⚠️  No se detectaron objetos en la imagen")
        print("   Opciones:")
        print("   1. Ajusta YOLO_CONF_THRESHOLD (actualmente: {:.2f})".format(YOLO_CONF_THRESHOLD))
        print("   2. Desactiva filtrar_solo_plantas=False")
        print("   3. Entrena YOLO específicamente con hojas de cacao")
        return None
    
    # ========================================
    # PASO 2: CLASIFICAR CADA HOJA DETECTADA
    # ========================================
    
    print("\n🧪 PASO 2: Clasificando deficiencias en cada hoja...")
    
    resultados_finales = []
    
    for idx, hoja in enumerate(hojas_detectadas):
        x1, y1, x2, y2 = hoja['bbox']
        
        # Recortar hoja
        hoja_recortada = img_cv[y1:y2, x1:x2]
        
        # Validar que el recorte no esté vacío
        if hoja_recortada.size == 0:
            continue
        
        # Redimensionar para el clasificador
        hoja_pil = Image.fromarray(hoja_recortada)
        hoja_resized = hoja_pil.resize((IMG_SIZE, IMG_SIZE))
        hoja_array = np.array(hoja_resized) / 255.0
        hoja_array = np.expand_dims(hoja_array, axis=0)
        
        # Clasificar
        predicciones = clasificador.predict(hoja_array, verbose=0)
        clase_pred = np.argmax(predicciones[0])
        confianza_clasificacion = predicciones[0][clase_pred] * 100
        categoria = CATEGORIAS[clase_pred]
        
        # Validación
        es_valido = True
        tipo_alerta = "success"
        
        if categoria == 'No-Cacao':
            es_valido = False
            tipo_alerta = "no-cacao"
        elif confianza_clasificacion < CONFIANZA_MINIMA * 100:
            es_valido = False
            tipo_alerta = "baja-confianza"
        
        # Guardar resultado
        resultado = {
            'id': idx + 1,
            'bbox': hoja['bbox'],
            'confianza_deteccion': hoja['confianza_deteccion'],
            'es_valido': es_valido,
            'categoria': categoria,
            'confianza_clasificacion': confianza_clasificacion,
            'probabilidades': {cat: float(predicciones[0][i]*100) 
                             for i, cat in enumerate(CATEGORIAS)},
            'tipo_alerta': tipo_alerta,
            'area_pixels': hoja['area']
        }
        
        resultados_finales.append(resultado)
        
        # Mostrar resultado individual
        print(f"\n   Hoja #{idx+1}:")
        print(f"      📍 Posición: ({x1}, {y1}) → ({x2}, {y2})")
        print(f"      🎯 Detección: {hoja['confianza_deteccion']:.1f}%")
        
        if es_valido:
            print(f"      ✅ Deficiencia: {categoria}")
            print(f"      📊 Confianza: {confianza_clasificacion:.1f}%")
        else:
            print(f"      ⚠️  ALERTA: {tipo_alerta}")
            print(f"      🔍 Clasificación tentativa: {categoria} ({confianza_clasificacion:.1f}%)")
    
    # ========================================
    # PASO 3: VISUALIZACIÓN
    # ========================================
    
    if mostrar_resultados or guardar_resultados:
        print("\n📊 Generando visualización...")
        
        # Calcular layout
        n_hojas = len(resultados_finales)
        n_cols = min(3, n_hojas)
        n_rows = (n_hojas + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(18, 6 + 4*n_rows))
        
        # Grid: [imagen principal] [hojas individuales]
        gs = fig.add_gridspec(n_rows + 1, n_cols, 
                             height_ratios=[2] + [1]*n_rows,
                             hspace=0.4, wspace=0.3)
        
        # ===== IMAGEN PRINCIPAL CON BOUNDING BOXES =====
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.imshow(img_cv)
        ax_main.axis('off')
        ax_main.set_title('Detecciones en Imagen Original', 
                         fontsize=16, fontweight='bold', pad=15)
        
        # Dibujar bounding boxes
        for res in resultados_finales:
            x1, y1, x2, y2 = res['bbox']
            
            # Color según validación
            if res['es_valido']:
                color = 'lime'
                label_prefix = '✓'
            else:
                color = 'red'
                label_prefix = '✗'
            
            # Box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=3, edgecolor=color,
                                     facecolor='none')
            ax_main.add_patch(rect)
            
            # Label
            label = f"{label_prefix} #{res['id']}: {res['categoria']}\n{res['confianza_clasificacion']:.1f}%"
            ax_main.text(x1, y1-10, label,
                        bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=color, alpha=0.8),
                        fontsize=10, fontweight='bold', color='black')
        
        # ===== HOJAS INDIVIDUALES =====
        for idx, res in enumerate(resultados_finales):
            row = 1 + idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            x1, y1, x2, y2 = res['bbox']
            hoja_img = img_cv[y1:y2, x1:x2]
            
            ax.imshow(hoja_img)
            ax.axis('off')
            
            # Título con resultado
            if res['es_valido']:
                titulo = f"✅ Hoja #{res['id']}: {res['categoria']}"
                color_titulo = 'green'
            else:
                titulo = f"⚠️ Hoja #{res['id']}: {res['tipo_alerta']}"
                color_titulo = 'red'
            
            ax.set_title(f"{titulo}\nConf: {res['confianza_clasificacion']:.1f}%",
                        fontsize=11, fontweight='bold', 
                        color=color_titulo, pad=8)
        
        plt.tight_layout()
        
        # Guardar
        if guardar_resultados:
            nombre_base = os.path.splitext(os.path.basename(ruta_imagen))[0]
            ruta_salida = os.path.join(BASE_PATH, f'resultado_{nombre_base}.png')
            plt.savefig(ruta_salida, dpi=200, bbox_inches='tight')
            print(f"   ✓ Resultado guardado: {ruta_salida}")
        
        if mostrar_resultados:
            plt.show()
        else:
            plt.close()
    
    # ========================================
    # RESUMEN FINAL
    # ========================================
    
    print("\n" + "="*70)
    print("📋 RESUMEN DE ANÁLISIS")
    print("="*70)
    print(f"🌿 Total de hojas detectadas: {len(resultados_finales)}")
    
    # Contar por categoría
    conteo = {}
    validas = 0
    invalidas = 0
    
    for res in resultados_finales:
        cat = res['categoria']
        conteo[cat] = conteo.get(cat, 0) + 1
        
        if res['es_valido']:
            validas += 1
        else:
            invalidas += 1
    
    print(f"✅ Hojas válidas: {validas}")
    print(f"⚠️  Hojas dudosas/inválidas: {invalidas}")
    
    print("\n📊 Distribución de deficiencias:")
    for cat, num in sorted(conteo.items(), key=lambda x: x[1], reverse=True):
        emoji = "🌿" if cat != "No-Cacao" else "❌"
        print(f"   {emoji} {cat}: {num} hoja(s)")
    
    print("="*70)
    
    return {
        'imagen': os.path.basename(ruta_imagen),
        'total_hojas': len(resultados_finales),
        'hojas_validas': validas,
        'hojas_invalidas': invalidas,
        'distribuciones': conteo,
        'detalles': resultados_finales
    }

# ========================================
# FUNCIÓN: PROCESAR LOTE DE IMÁGENES
# ========================================

def procesar_lote(directorio_imagenes, extension='.jpg'):
    """
    Procesa todas las imágenes en un directorio
    """
    if not os.path.exists(directorio_imagenes):
        print(f"❌ No existe el directorio: {directorio_imagenes}")
        return
    
    imagenes = [f for f in os.listdir(directorio_imagenes) 
                if f.lower().endswith(extension.lower())]
    
    if not imagenes:
        print(f"❌ No se encontraron imágenes {extension} en {directorio_imagenes}")
        return
    
    print(f"\n🔄 Procesando {len(imagenes)} imágenes...\n")
    
    resultados_totales = []
    
    for img in imagenes:
        ruta = os.path.join(directorio_imagenes, img)
        resultado = detectar_y_clasificar_hojas(ruta, 
                                                mostrar_resultados=False,
                                                guardar_resultados=True)
        if resultado:
            resultados_totales.append(resultado)
    
    # Resumen general
    print("\n" + "="*70)
    print("📊 RESUMEN GENERAL DEL LOTE")
    print("="*70)
    
    total_imgs = len(resultados_totales)
    total_hojas = sum(r['total_hojas'] for r in resultados_totales)
    
    print(f"📁 Imágenes procesadas: {total_imgs}")
    print(f"🌿 Total de hojas detectadas: {total_hojas}")
    print(f"📈 Promedio: {total_hojas/total_imgs:.1f} hojas por imagen")
    
    return resultados_totales

# ========================================
# MODO DE USO
# ========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("📖 CÓMO USAR ESTE DETECTOR")
    print("="*70)
    
    print("\n1️⃣  PROCESAR UNA IMAGEN:")
    print("    resultado = detectar_y_clasificar_hojas(r'C:\\ruta\\imagen.jpg')")
    
    print("\n2️⃣  PROCESAR SIN FILTRO DE PLANTAS:")
    print("    detectar_y_clasificar_hojas(r'imagen.jpg', filtrar_solo_plantas=False)")
    
    print("\n3️⃣  PROCESAR LOTE DE IMÁGENES:")
    print("    procesar_lote(r'C:\\carpeta\\imagenes')")
    
    print("\n4️⃣  AJUSTAR PARÁMETROS:")
    print("    # Al inicio del script, modifica:")
    print("    YOLO_CONF_THRESHOLD = 0.25  # Sensibilidad de detección")
    print("    CONFIANZA_MINIMA = 0.60     # Umbral clasificación")
    
    print("\n" + "="*70)
    print("🎯 MEJORANDO LA PRECISIÓN: ENTRENAR YOLO CON TUS DATOS")
    print("="*70)
    
    print("\nPara detectar hojas de cacao específicamente:")
    print("\n1. Anota tus imágenes (usa LabelImg, CVAT o Roboflow)")
    print("2. Crea dataset en formato YOLO:")
    print("   dataset/")
    print("     ├── images/")
    print("     │   ├── train/")
    print("     │   └── val/")
    print("     ├── labels/")
    print("     │   ├── train/")
    print("     │   └── val/")
    print("     └── data.yaml")
    
    print("\n3. Entrena YOLO:")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('yolov8n.pt')")
    print("   model.train(data='dataset/data.yaml', epochs=50)")
    
    print("\n4. Usa tu modelo entrenado:")
    print("   detector_yolo = YOLO('runs/detect/train/weights/best.pt')")
    
    print("\n💡 TIPS:")
    print("   • Necesitas mínimo 100-200 imágenes anotadas")
    print("   • Varía ángulos, iluminación y fondos")
    print("   • Anota cada hoja individual en imágenes con múltiples hojas")
    
    print("\n" + "="*70)
    print("\n🚀 ¡Sistema listo! Ejecuta las funciones para empezar.")
    print("\n" + "="*70)