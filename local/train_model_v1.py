# ========================================
# MODELO DE DETECCIÓN DE DEFICIENCIAS NUTRICIONALES EN HOJAS DE CACAO
# Versión para LOCALHOST (Windows/Linux/Mac)
# Categorías: Potasio, Nitrógeno, Fósforo
# ========================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir warnings de TensorFlow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURACIÓN DE RUTAS - LOCALHOST
# ========================================

# Ruta base donde está tu proyecto
BASE_PATH = r'C:\Users\User\Desktop\local'

# Ruta del dataset (debe contener carpetas: Potasio, Nitrogeno, Fosforo)
DATASET_PATH = os.path.join(BASE_PATH, 'dataset_cacao')

# Ruta donde se guardarán los modelos y resultados
MODELS_PATH = BASE_PATH
RESULTS_PATH = BASE_PATH

# Crear directorios si no existen
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

print("="*60)
print("🌿 DETECTOR DE DEFICIENCIAS NUTRICIONALES EN CACAO")
print("="*60)

# ========================================
# CONFIGURACIÓN DEL MODELO
# ========================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

categorias = ['Potasio', 'Nitrogeno', 'Fosforo']
num_classes = len(categorias)

print(f"\n📊 Configuración:")
print(f"   • Categorías: {categorias}")
print(f"   • Tamaño de imagen: {IMG_SIZE}x{IMG_SIZE}")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Epochs: {EPOCHS}")

# Verificar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   • GPU: ✅ Disponible ({len(gpus)} dispositivo(s))")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"   • Advertencia GPU: {e}")
else:
    print("   • GPU: ❌ No disponible (usando CPU)")

# ========================================
# VERIFICAR DATASET
# ========================================

print(f"\n🔍 Verificando dataset en: {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERROR: No se encuentra el directorio del dataset")
    print(f"   Por favor, crea la carpeta: {DATASET_PATH}")
    print(f"   Y dentro crea las subcarpetas: Potasio, Nitrogeno, Fosforo")
    exit()

total_imagenes = 0
for categoria in categorias:
    path = os.path.join(DATASET_PATH, categoria)
    if os.path.exists(path):
        imagenes = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_imgs = len(imagenes)
        total_imagenes += num_imgs
        print(f"   ✓ {categoria}: {num_imgs} imágenes")
    else:
        print(f"   ✗ {categoria}: CARPETA NO ENCONTRADA")
        print(f"      Crea la carpeta: {path}")
        exit()

if total_imagenes == 0:
    print("\n❌ ERROR: No se encontraron imágenes en el dataset")
    print("   Asegúrate de tener imágenes (.jpg, .jpeg, .png) en las carpetas")
    exit()

print(f"\n✅ Total de imágenes encontradas: {total_imagenes}")

# ========================================
# PREPARACIÓN DE DATOS CON AUGMENTACIÓN
# ========================================

print("\n🔄 Preparando generadores de datos...")

# Generador para entrenamiento (con augmentación)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.15,
    fill_mode='nearest'
)

# Generador para validación (sin augmentación)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Crear generadores
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"   • Entrenamiento: {train_generator.samples} imágenes")
print(f"   • Validación: {validation_generator.samples} imágenes")

# ========================================
# CREAR MODELO CON TRANSFER LEARNING
# ========================================

print("\n🏗️ Construyendo modelo con MobileNetV2...")

# Cargar modelo base pre-entrenado
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Congelar capas del modelo base
base_model.trainable = False

# Construir modelo completo
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compilar modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"   • Parámetros totales: {model.count_params():,}")
print(f"   • Parámetros entrenables: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ========================================
# CALLBACKS
# ========================================

mejor_modelo_path = os.path.join(MODELS_PATH, 'mejor_modelo_cacao.h5')
modelo_final_path = os.path.join(MODELS_PATH, 'modelo_final_cacao.h5')
resultados_img_path = os.path.join(RESULTS_PATH, 'resultados_entrenamiento_cacao.png')

callbacks = [
    keras.callbacks.ModelCheckpoint(
        mejor_modelo_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
]

# ========================================
# ENTRENAR MODELO
# ========================================

print("\n🚀 Iniciando entrenamiento...")
print("="*60)

inicio = datetime.now()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

tiempo_total = datetime.now() - inicio

print("="*60)
print(f"⏱️ Tiempo total: {tiempo_total}")

# ========================================
# VISUALIZAR RESULTADOS
# ========================================

print("\n📊 Generando gráficas...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, marker='o')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, marker='s')
ax1.set_title('Precisión del Modelo', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Train Loss', linewidth=2, marker='o')
ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2, marker='s')
ax2.set_title('Pérdida del Modelo', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(resultados_img_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Gráficas guardadas en: {resultados_img_path}")
plt.show()

# ========================================
# EVALUACIÓN FINAL
# ========================================

print("\n🎯 Evaluación final:")
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"   • Pérdida: {val_loss:.4f}")
print(f"   • Precisión: {val_accuracy*100:.2f}%")

# ========================================
# GUARDAR MODELO FINAL
# ========================================

print("\n💾 Guardando modelo final...")
model.save(modelo_final_path)
print(f"   ✓ Modelo guardado en: {modelo_final_path}")

# ========================================
# FUNCIÓN DE PREDICCIÓN
# ========================================

def predecir_deficiencia(ruta_imagen, mostrar_grafica=True):
    """
    Predice la deficiencia nutricional de una hoja de cacao
    
    Args:
        ruta_imagen: Ruta a la imagen de la hoja
        mostrar_grafica: Si True, muestra visualización
    
    Returns:
        tuple: (categoría_predicha, probabilidades, confianza)
    """
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: No se encuentra la imagen en {ruta_imagen}")
        return None, None, None
    
    # Cargar y preprocesar imagen
    img = keras.preprocessing.image.load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predecir
    predicciones = model.predict(img_array, verbose=0)
    clase_predicha = np.argmax(predicciones[0])
    confianza = predicciones[0][clase_predicha] * 100
    
    if mostrar_grafica:
        # Mostrar imagen y resultados
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Imagen Analizada', fontsize=12, fontweight='bold')
        
        plt.subplot(1, 2, 2)
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        barras = plt.barh(categorias, predicciones[0] * 100, color=colores)
        plt.xlabel('Probabilidad (%)', fontsize=11)
        plt.title('Predicciones', fontsize=12, fontweight='bold')
        plt.xlim(0, 100)
        
        for i, v in enumerate(predicciones[0] * 100):
            plt.text(v + 2, i, f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    # Imprimir resultados
    print(f"\n🌿 RESULTADO:")
    print(f"   • Deficiencia detectada: {categorias[clase_predicha]}")
    print(f"   • Confianza: {confianza:.2f}%")
    print(f"\n📊 Probabilidades detalladas:")
    for i, cat in enumerate(categorias):
        print(f"   • {cat}: {predicciones[0][i]*100:.2f}%")
    
    return categorias[clase_predicha], predicciones[0], confianza

# ========================================
# RESUMEN FINAL
# ========================================

print("\n" + "="*60)
print("✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
print("="*60)

print(f"\n📂 Archivos generados en: {BASE_PATH}")
print(f"   • {mejor_modelo_path}")
print(f"   • {modelo_final_path}")
print(f"   • {resultados_img_path}")

print("\n📝 CÓMO USAR EL MODELO:")
print(f"\n1. Para predecir una imagen:")
print(f"   predecir_deficiencia(r'C:\\ruta\\a\\tu\\imagen.jpg')")

print(f"\n2. Ejemplo con una imagen del dataset:")
print(f"   predecir_deficiencia(r'{DATASET_PATH}\\Potasio\\imagen1.jpg')")

print("\n3. Para usar el modelo en otro script:")
print(f"   model = keras.models.load_model(r'{modelo_final_path}')")

print("\n" + "="*60)

# ========================================
# PRUEBA CON IMAGEN DE VALIDACIÓN
# ========================================

print("\n🧪 Probando con una imagen del dataset de validación...\n")

try:
    x_val, y_val = next(validation_generator)
    img_test = x_val[0:1]
    pred = model.predict(img_test, verbose=0)
    clase_real = np.argmax(y_val[0])
    clase_pred = np.argmax(pred[0])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img_test[0])
    plt.axis('off')
    
    titulo = f'Real: {categorias[clase_real]} | Predicción: {categorias[clase_pred]}\n'
    titulo += f'Confianza: {pred[0][clase_pred]*100:.1f}%'
    
    if clase_real == clase_pred:
        titulo = '✅ ' + titulo
        color = 'green'
    else:
        titulo = '❌ ' + titulo
        color = 'red'
    
    plt.title(titulo, fontsize=12, fontweight='bold', color=color)
    plt.tight_layout()
    plt.show()
    
    print("🎉 ¡Modelo listo para usar!")
    
except Exception as e:
    print(f"⚠️ No se pudo realizar la prueba: {e}")
    print("   Pero el modelo está entrenado y guardado correctamente.")

print("\n" + "="*60)