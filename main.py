from fastapi import FastAPI, File, UploadFile
#from tensorflow.keras.models import load_model
from keras.saving import load_model
from PIL import Image
import numpy as np
import io

# ======================================
# API PARA DETECCIÓN DE DEFICIENCIAS EN CACAO
# CLASES: Potasio, Nitrogeno, Fosforo
# ======================================

app = FastAPI(title="Modelo Cacao", description="Clasificador de deficiencias nutricionales en hojas de cacao")

print("🔄 Cargando modelo...")
model = load_model(
    "modelo_final_cacao.h5",
    compile=False,          # evita intentar restaurar Optimizador/Sesión vieja
    safe_mode=False         # permite cargar capas Legacy (como InputLayer)
)
print("Modelo cargado correctamente ✔")

CLASS_NAMES = ["Potasio", "Nitrogeno", "Fosforo"]
IMG_SIZE = 224  # Tamaño usado en MobileNetV2

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.get("/")
def home():
    return {"status": "API activa 🌱", "clases": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_data = preprocess_image(img)
    predictions = model.predict(input_data)[0]
    index = np.argmax(predictions)

    return {
        "clase_predicha": CLASS_NAMES[index],
        "confianza": float(predictions[index] * 100),
        "probabilidades": {
            CLASS_NAMES[i]: float(predictions[i] * 100) for i in range(3)
        }
    }

