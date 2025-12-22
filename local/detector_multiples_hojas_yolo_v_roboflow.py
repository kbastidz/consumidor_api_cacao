import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# ========================================
# CONFIGURACIÓN
# ========================================

YOLO_MODEL_PATH = r'runs/detect/cacao_detector/weights/best.pt'
OUTPUT_DIR = "resultados"

CONF_THRESHOLD = 0.15   # bájalo si quieres más cajas
IOU_THRESHOLD = 0.60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# CARGA DEL MODELO
# ========================================

modelo = YOLO(YOLO_MODEL_PATH)
CLASES = modelo.names

print("✓ Clases del modelo:", CLASES)

# ========================================
# FUNCIÓN PRINCIPAL
# ========================================

def detectar_deficiencias(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print("❌ Imagen no encontrada")
        return

    img = cv2.imread(ruta_imagen)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inferencia
    results = modelo(
        img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print("⚠️ No se detectó nada")
        return

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() * 100
    clases = boxes.cls.cpu().numpy().astype(int)

    # ========================================
    # DIBUJO
    # ========================================

    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img)
    ax.axis("off")

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        clase_id = clases[i]
        nombre = CLASES[clase_id]
        confianza = confs[i]

        label = f"{nombre} ({confianza:.1f}%)"

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=3,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 8,
            label,
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="lime", alpha=0.75)
        )

    # ========================================
    # GUARDAR RESULTADO
    # ========================================

    out_path = os.path.join(
        OUTPUT_DIR,
        f"resultado_{os.path.basename(ruta_imagen)}"
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"💾 Imagen generada: {out_path}")

# ========================================
# EJECUCIÓN
# ========================================

if __name__ == "__main__":
    #detectar_deficiencias("test_v1.jpeg")
    detectar_deficiencias("noimagen.png")
