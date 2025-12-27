from ultralytics import YOLO

# Cargar tu modelo ya entrenado
YOLO_MODEL_PATH = r"C:\Users\User\Documents\GitHub\consumidor_api_cacao\local\fase_final\runs\detect\train2\weights\best.pt"

model = YOLO(YOLO_MODEL_PATH)

# Verificación con una imagen
results = model.predict(
    source="ms2.jpg",   # aquí pones la imagen que quieres probar
    save=True,
    conf=0.7
)

#que rico hp

print("Predicción completada. Revisa runs/detect/predict/")
