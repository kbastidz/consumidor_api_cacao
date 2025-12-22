from ultralytics import YOLO

model = YOLO("runs/detect/cacao_detector/weights/best.pt")

print("Clases del modelo:", model.names)

results = model(
    r'C:\Users\User\Documents\GitHub\consumidor_api_cacao\local\dataset_cacao\Potasio\test_v1.jpeg',
    conf=0.001,   # ULTRA bajo
    iou=0.5,
    save=True
)

print("Boxes:", results[0].boxes)
