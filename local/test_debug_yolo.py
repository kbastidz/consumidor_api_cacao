from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Cargar tu modelo
model = YOLO('runs/detect/cacao_detector/weights/best.pt')

# Probar con una imagen
imagen = r'C:\Users\User\Documents\GitHub\consumidor_api_cacao\local\dataset_cacao\Potasio\prueba.webp'

# Detectar con umbral muy bajo para ver TODO
results = model(imagen, conf=0.01, verbose=True)

# Mostrar resultado
results[0].plot()
plt.show()

# Ver qué detectó
print("\nDetecciones:")
for box in results[0].boxes:
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    print(f"   Clase: {model.names[cls]}, Confianza: {conf:.2%}")