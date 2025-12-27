from ultralytics import YOLO

# Cargar modelo base
model = YOLO("yolov8n.pt")

# Entrenar con solo 4 epochs
model.train(
    data="data.yaml",
    epochs=250, #va explotar xd bello
    imgsz=640,
    batch=16,
    workers=8,
)

# Verificación con una imagen
results = model.predict(
    source="hoja.jpg",
    save=True,
    conf=0.25
)

print("Predicción completada. Revisa runs/detect/predict/")
