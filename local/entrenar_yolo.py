from ultralytics import YOLO
import os

# Configuración
DATASET_PATH = r'C:\Users\User\Documents\GitHub\consumidor_api_cacao\local\dataset_yolo_cacao\data.yaml'
EPOCHS = 50          # Número de épocas (ajusta según tu tiempo)
IMG_SIZE = 640       # Tamaño de imagen
BATCH_SIZE = 16      # Ajusta según tu GPU (8 si da error de memoria)

print("="*70)
print("🚀 ENTRENANDO YOLO PARA DETECCIÓN DE HOJAS DE CACAO")
print("="*70)

# Verificar que existe el dataset
if not os.path.exists(DATASET_PATH):
    print(f"❌ No se encuentra: {DATASET_PATH}")
    exit()

print(f"\n📊 Configuración:")
print(f"   • Dataset: {DATASET_PATH}")
print(f"   • Épocas: {EPOCHS}")
print(f"   • Tamaño imagen: {IMG_SIZE}")
print(f"   • Batch size: {BATCH_SIZE}")

# Cargar modelo pre-entrenado (Transfer Learning)
print("\n📦 Cargando YOLOv8 nano...")
model = YOLO('yolov8n.pt')  # Modelo más ligero y rápido

# Entrenar
print("\n🏋️  Iniciando entrenamiento...")
print("   (Esto puede tardar 30min - 2h según tu hardware)\n")

results = model.train(
    data=DATASET_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name='cacao_detector',
    patience=10,          # Early stopping
    save=True,
    plots=True,
    device='cpu',          # Usa GPU si está disponible 0, sino cambia a 'cpu'
    exist_ok=True
)

print("\n" + "="*70)
print("✅ ¡ENTRENAMIENTO COMPLETADO!")
print("="*70)

print("\n📂 Modelo guardado en:")
print("   runs/detect/cacao_detector/weights/best.pt")

print("\n🎯 Para usar tu modelo entrenado:")
print("   Edita 'detector_multiples_hojas_yolo.py' línea 66:")
print("   detector_yolo = YOLO('runs/detect/cacao_detector/weights/best.pt')")