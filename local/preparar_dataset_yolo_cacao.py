# ========================================
# PREPARACIÓN Y ENTRENAMIENTO DE YOLO PARA HOJAS DE CACAO
# ========================================

import os
import shutil
from pathlib import Path
import yaml
import random
from PIL import Image

# ========================================
# CONFIGURACIÓN
# ========================================

BASE_PATH = r'C:\Users\User\Documents\GitHub\consumidor_api_cacao\local'
DATASET_YOLO = os.path.join(BASE_PATH, 'dataset_yolo_cacao')

# Crear estructura
IMAGES_TRAIN = os.path.join(DATASET_YOLO, 'images', 'train')
IMAGES_VAL = os.path.join(DATASET_YOLO, 'images', 'val')
LABELS_TRAIN = os.path.join(DATASET_YOLO, 'labels', 'train')
LABELS_VAL = os.path.join(DATASET_YOLO, 'labels', 'val')

print("="*70)
print("🎯 PREPARACIÓN DE DATASET YOLO PARA HOJAS DE CACAO")
print("="*70)

# ========================================
# PASO 1: CREAR ESTRUCTURA DE CARPETAS
# ========================================

def crear_estructura():
    """Crea la estructura de carpetas necesaria para YOLO"""
    
    print("\n📁 Creando estructura de carpetas...")
    
    for carpeta in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        os.makedirs(carpeta, exist_ok=True)
        print(f"   ✓ {carpeta}")
    
    print("\n✅ Estructura creada")

# ========================================
# PASO 2: CREAR ARCHIVO data.yaml
# ========================================

def crear_data_yaml():
    """Crea el archivo de configuración para YOLO"""
    
    data_yaml = {
        'path': DATASET_YOLO.replace('\\', '/'),  # Ruta absoluta
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Número de clases (solo "hoja de cacao")
        'names': ['cacao_leaf']  # Nombres de las clases
    }
    
    yaml_path = os.path.join(DATASET_YOLO, 'data.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n📄 Archivo data.yaml creado en: {yaml_path}")
    print("\n   Contenido:")
    print(f"   • Clases: {data_yaml['names']}")
    print(f"   • Train: {data_yaml['train']}")
    print(f"   • Val: {data_yaml['val']}")
    
    return yaml_path

# ========================================
# PASO 3: FUNCIÓN PARA CONVERTIR ANOTACIONES
# ========================================

def convertir_a_formato_yolo(x_center, y_center, width, height, img_width, img_height):
    """
    Convierte coordenadas absolutas a formato YOLO (normalizado)
    
    Args:
        x_center, y_center: Centro del bounding box (píxeles)
        width, height: Dimensiones del bounding box (píxeles)
        img_width, img_height: Dimensiones de la imagen
    
    Returns:
        Coordenadas YOLO normalizadas (0-1)
    """
    
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

# ========================================
# PASO 4: HERRAMIENTA PARA ANOTAR IMÁGENES
# ========================================

def crear_anotacion_ejemplo(imagen_path, bbox_coords, output_path):
    """
    Crea un archivo de anotación YOLO
    
    Args:
        imagen_path: Ruta de la imagen
        bbox_coords: Lista de tuplas (x_center, y_center, width, height) en píxeles
        output_path: Ruta donde guardar el .txt
    """
    
    # Obtener dimensiones de la imagen
    img = Image.open(imagen_path)
    img_width, img_height = img.size
    
    # Crear archivo de anotación
    with open(output_path, 'w') as f:
        for coords in bbox_coords:
            x_c, y_c, w, h = coords
            
            # Convertir a formato YOLO
            x_norm, y_norm, w_norm, h_norm = convertir_a_formato_yolo(
                x_c, y_c, w, h, img_width, img_height
            )
            
            # Formato YOLO: clase x_center y_center width height
            f.write(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    print(f"   ✓ Anotación guardada: {output_path}")

# ========================================
# PASO 5: DIVIDIR DATASET EN TRAIN/VAL
# ========================================

def dividir_dataset(imagenes_anotadas, split_ratio=0.8):
    """
    Divide las imágenes anotadas en train y validation
    
    Args:
        imagenes_anotadas: Lista de tuplas (ruta_imagen, ruta_label)
        split_ratio: Proporción para entrenamiento (0.8 = 80% train, 20% val)
    """
    
    random.shuffle(imagenes_anotadas)
    
    split_idx = int(len(imagenes_anotadas) * split_ratio)
    train_data = imagenes_anotadas[:split_idx]
    val_data = imagenes_anotadas[split_idx:]
    
    print(f"\n📊 División del dataset:")
    print(f"   • Train: {len(train_data)} imágenes ({split_ratio*100:.0f}%)")
    print(f"   • Val: {len(val_data)} imágenes ({(1-split_ratio)*100:.0f}%)")
    
    # Copiar archivos
    print("\n🔄 Copiando archivos...")
    
    for img_path, label_path in train_data:
        shutil.copy(img_path, IMAGES_TRAIN)
        shutil.copy(label_path, LABELS_TRAIN)
    
    for img_path, label_path in val_data:
        shutil.copy(img_path, IMAGES_VAL)
        shutil.copy(label_path, LABELS_VAL)
    
    print("   ✅ Archivos copiados")

# ========================================
# PASO 6: ENTRENAR YOLO
# ========================================

def entrenar_yolo(data_yaml_path, epochs=50, img_size=640, batch=16):
    """
    Entrena YOLOv8 con el dataset preparado
    
    Args:
        data_yaml_path: Ruta al archivo data.yaml
        epochs: Número de épocas
        img_size: Tamaño de imagen
        batch: Tamaño del batch
    """
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ Instala ultralytics: pip install ultralytics")
        return
    
    print("\n" + "="*70)
    print("🚀 INICIANDO ENTRENAMIENTO DE YOLO")
    print("="*70)
    
    print(f"\n⚙️  Parámetros:")
    print(f"   • Épocas: {epochs}")
    print(f"   • Tamaño de imagen: {img_size}")
    print(f"   • Batch size: {batch}")
    
    # Cargar modelo pre-entrenado
    model = YOLO('yolov8n.pt')  # nano (más rápido)
    # Otras opciones: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    
    print("\n📦 Modelo base: YOLOv8n (nano)")
    
    # Entrenar
    print("\n🏋️  Entrenando...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name='cacao_leaf_detector',
        patience=10,  # Early stopping
        save=True,
        plots=True,
        device='0' if os.system('nvidia-smi') == 0 else 'cpu'  # GPU si está disponible
    )
    
    print("\n" + "="*70)
    print("✅ ¡ENTRENAMIENTO COMPLETADO!")
    print("="*70)
    
    print("\n📂 Archivos generados en: runs/detect/cacao_leaf_detector/")
    print("   • weights/best.pt  →  Mejor modelo")
    print("   • weights/last.pt  →  Último checkpoint")
    print("   • Gráficas de entrenamiento")
    
    print("\n💡 Para usar tu modelo entrenado:")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('runs/detect/cacao_leaf_detector/weights/best.pt')")
    print("   results = model('imagen.jpg')")
    
    return results

# ========================================
# GUÍA DE USO COMPLETA
# ========================================

def mostrar_guia_completa():
    """Muestra la guía paso a paso para preparar el dataset"""
    
    print("\n" + "="*70)
    print("📚 GUÍA COMPLETA: CÓMO PREPARAR TU DATASET YOLO")
    print("="*70)
    
    print("\n" + "▶"*35)
    print("OPCIÓN 1: ANOTAR CON HERRAMIENTAS GRÁFICAS (RECOMENDADO)")
    print("▶"*35)
    
    print("\n🔧 Herramientas de anotación:")
    
    print("\n1️⃣  LABELIMG (Más simple, offline)")
    print("   • Instalar: pip install labelImg")
    print("   • Ejecutar: labelImg")
    print("   • Formato: Selecciona 'YOLO' en el menú")
    print("   • Anotar: Dibuja cajas alrededor de cada hoja")
    print("   • Guarda automáticamente los .txt en formato YOLO")
    
    print("\n2️⃣  ROBOFLOW (Más potente, online)")
    print("   • Web: https://roboflow.com")
    print("   • Sube tus imágenes")
    print("   • Anota cada hoja con bounding boxes")
    print("   • Exporta en formato 'YOLOv8'")
    print("   • Descarga y descomprime en dataset_yolo_cacao/")
    
    print("\n3️⃣  CVAT (Profesional, auto-anotación)")
    print("   • Web: https://app.cvat.ai")
    print("   • Soporte para auto-anotación con IA")
    print("   • Exporta en formato YOLO")
    
    print("\n" + "▶"*35)
    print("OPCIÓN 2: ANOTAR PROGRAMÁTICAMENTE")
    print("▶"*35)
    
    print("\nSi prefieres código Python:")
    print("""
# Ejemplo: Anotar una imagen con 2 hojas
from PIL import Image

imagen = 'mi_imagen.jpg'
img = Image.open(imagen)
width, height = img.size

# Coordenadas de cada hoja (en píxeles)
# (x_center, y_center, width_box, height_box)
hojas = [
    (300, 200, 150, 200),  # Hoja 1
    (500, 400, 180, 220),  # Hoja 2
]

# Convertir y guardar
crear_anotacion_ejemplo(
    imagen, 
    hojas, 
    'mi_imagen.txt'
)
""")
    
    print("\n" + "▶"*35)
    print("ESTRUCTURA FINAL DEL DATASET")
    print("▶"*35)
    
    print("""
dataset_yolo_cacao/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val/
│       ├── img100.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    └── val/
        ├── img100.txt
        └── ...

Cada .txt contiene líneas con formato YOLO:
clase x_center y_center width height
0 0.516 0.612 0.248 0.356
0 0.234 0.445 0.312 0.289
(Todos los valores normalizados 0-1)
""")
    
    print("\n" + "▶"*35)
    print("RECOMENDACIONES PARA BUENOS RESULTADOS")
    print("▶"*35)
    
    print("\n📊 Cantidad de datos:")
    print("   • Mínimo: 100-200 imágenes anotadas")
    print("   • Recomendado: 500-1000 imágenes")
    print("   • Óptimo: 2000+ imágenes")
    
    print("\n🎨 Variedad:")
    print("   • Diferentes ángulos de las hojas")
    print("   • Varias condiciones de iluminación")
    print("   • Fondos variados (campo, laboratorio)")
    print("   • Hojas a diferentes distancias")
    print("   • Imágenes con 1 hoja y con múltiples hojas")
    
    print("\n✅ Calidad de anotaciones:")
    print("   • Ajusta bien las cajas a los bordes de la hoja")
    print("   • No dejes hojas sin anotar")
    print("   • Anota incluso hojas parcialmente visibles")
    print("   • Sé consistente en todas las imágenes")
    
    print("\n⚠️  Evitar:")
    print("   • Imágenes muy borrosas")
    print("   • Hojas muy pequeñas (< 20x20 píxeles)")
    print("   • Imágenes duplicadas")
    
    print("\n" + "="*70)

# ========================================
# FUNCIÓN PRINCIPAL
# ========================================

def preparar_dataset_completo():
    """Ejecuta todos los pasos de preparación"""
    
    crear_estructura()
    yaml_path = crear_data_yaml()
    
    print("\n" + "="*70)
    print("✅ DATASET PREPARADO")
    print("="*70)
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("\n1. Anota tus imágenes usando LabelImg, Roboflow o CVAT")
    print("2. Coloca las imágenes anotadas en:")
    print(f"   {IMAGES_TRAIN}")
    print(f"   {IMAGES_VAL}")
    print("3. Asegúrate que cada .jpg tenga su .txt correspondiente")
    print("4. Ejecuta el entrenamiento:")
    print("   entrenar_yolo(r'" + yaml_path + "')")
    
    return yaml_path

# ========================================
# EJEMPLOS DE USO
# ========================================

if __name__ == "__main__":
    
    mostrar_guia_completa()
    
    print("\n" + "="*70)
    print("🛠️  FUNCIONES DISPONIBLES")
    print("="*70)
    
    print("\n1️⃣  Preparar estructura del dataset:")
    print("   preparar_dataset_completo()")
    
    print("\n2️⃣  Entrenar YOLO (después de anotar):")
    print("   entrenar_yolo(")
    print("       data_yaml_path=r'" + os.path.join(DATASET_YOLO, 'data.yaml') + "',")
    print("       epochs=50,")
    print("       img_size=640,")
    print("       batch=16")
    print("   )")
    
    print("\n3️⃣  Crear anotación para una imagen:")
    print("   crear_anotacion_ejemplo(")
    print("       imagen_path='imagen.jpg',")
    print("       bbox_coords=[(300, 200, 150, 200)],  # x, y, w, h en píxeles")
    print("       output_path='imagen.txt'")
    print("   )")
    
    print("\n" + "="*70)
    print("\n💡 CONSEJO: Comienza con pocas imágenes (~50) para probar")
    print("   el proceso completo antes de anotar cientos de imágenes")
    print("\n" + "="*70)