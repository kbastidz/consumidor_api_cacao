# ========================================
# MODELO DE DETECCIÓN DE DEFICIENCIAS NUTRICIONALES EN HOJAS DE CACAO
# CON VALIDACIÓN DE HOJAS NO-CACAO
# Versión para LOCALHOST (Windows/Linux/Mac)
# Categorías: Potasio, Nitrógeno, Fósforo, No-Cacao
# ========================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

BASE_PATH = r'C:\Users\User\Desktop\local'
DATASET_PATH = os.path.join(BASE_PATH, 'dataset_cacao')
MODELS_PATH = BASE_PATH
RESULTS_PATH = BASE_PATH

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

print("="*60)
print("🌿 DETECTOR DE DEFICIENCIAS NUTRICIONALES EN CACAO")
print("   CON VALIDACIÓN DE IMÁGENES NO-CACAO")
print("="*60)

# ========================================
# CONFIGURACIÓN DEL MODELO
# ========================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# IMPORTANTE: Ahora incluimos la clase "No-Cacao"
categorias = ['Potasio', 'Nitrogeno', 'Fosforo', 'No-Cacao']
num_classes = len(categorias)

# Umbral de confianza para considerar una predicción válida
CONFIANZA_MINIMA = 0.60  # 60% - Ajustable según tus necesidades

print(f"\n📊 Configuración:")
print(f"   • Categorías: {categorias}")
print(f"   • Umbral de confianza mínima: {CONFIANZA_MINIMA*100}%")
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
    print(f"   Y dentro crea las subcarpetas: {', '.join(categorias)}")
    exit()

total_imagenes = 0
print("\n📁 Estructura del dataset requerida:")
print(f"   {DATASET_PATH}/")
for categoria in categorias:
    path = os.path.join(DATASET_PATH, categoria)
    if os.path.exists(path):
        imagenes = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_imgs = len(imagenes)
        total_imagenes += num_imgs
        print(f"   ├── {categoria}/  →  {num_imgs} imágenes ✓")
    else:
        print(f"   ├── {categoria}/  →  CARPETA NO ENCONTRADA ✗")
        if categoria == 'No-Cacao':
            print(f"       ⚠️  IMPORTANTE: Crea esta carpeta y agrega imágenes de:")
            print(f"           • Otras plantas (flores, otras hojas, césped)")
            print(f"           • Objetos (manos, herramientas, tierra)")
            print(f"           • Fondos (cielo, suelo, texturas)")
            print(f"           Mínimo recomendado: 50-100 imágenes variadas")
        exit()

if total_imagenes == 0:
    print("\n❌ ERROR: No se encontraron imágenes en el dataset")
    exit()

print(f"\n✅ Total de imágenes encontradas: {total_imagenes}")

# ========================================
# PREPARACIÓN DE DATOS CON AUGMENTACIÓN
# ========================================

print("\n🔄 Preparando generadores de datos...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

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
print(f"\n   Mapeo de clases: {train_generator.class_indices}")

# ========================================
# CREAR MODELO CON TRANSFER LEARNING
# ========================================

print("\n🏗️ Construyendo modelo con MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print(f"   • Parámetros totales: {model.count_params():,}")
print(f"   • Parámetros entrenables: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ========================================
# CALLBACKS
# ========================================

mejor_modelo_path = os.path.join(MODELS_PATH, 'mejor_modelo_cacao_validado.h5')
modelo_final_path = os.path.join(MODELS_PATH, 'modelo_final_cacao_validado.h5')
resultados_img_path = os.path.join(RESULTS_PATH, 'resultados_entrenamiento_cacao_validado.png')

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
        patience=8,
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

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2, marker='o')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s')
axes[0, 0].set_title('Precisión del Modelo', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2, marker='o')
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s')
axes[0, 1].set_title('Pérdida del Modelo', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2, marker='o')
axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2, marker='s')
axes[1, 0].set_title('Precisión (Precision)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2, marker='o')
axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2, marker='s')
axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(resultados_img_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Gráficas guardadas en: {resultados_img_path}")
plt.show()

# ========================================
# EVALUACIÓN FINAL
# ========================================

print("\n🎯 Evaluación final:")
resultados = model.evaluate(validation_generator, verbose=0)
print(f"   • Pérdida: {resultados[0]:.4f}")
print(f"   • Precisión (Accuracy): {resultados[1]*100:.2f}%")
print(f"   • Precisión (Precision): {resultados[2]*100:.2f}%")
print(f"   • Recall: {resultados[3]*100:.2f}%")

# ========================================
# GUARDAR MODELO FINAL
# ========================================

print("\n💾 Guardando modelo final...")
model.save(modelo_final_path)
print(f"   ✓ Modelo guardado en: {modelo_final_path}")

# ========================================
# FUNCIÓN DE PREDICCIÓN CON VALIDACIÓN
# ========================================

def predecir_deficiencia(ruta_imagen, mostrar_grafica=True, umbral_confianza=CONFIANZA_MINIMA):
    """
    Predice la deficiencia nutricional de una hoja de cacao con validación
    
    Args:
        ruta_imagen: Ruta a la imagen de la hoja
        mostrar_grafica: Si True, muestra visualización
        umbral_confianza: Umbral mínimo de confianza (0-1)
    
    Returns:
        dict: Diccionario con los resultados de la predicción
    """
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: No se encuentra la imagen en {ruta_imagen}")
        return None
    
    # Cargar y preprocesar imagen
    img = keras.preprocessing.image.load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predecir
    predicciones = model.predict(img_array, verbose=0)
    clase_predicha = np.argmax(predicciones[0])
    confianza = predicciones[0][clase_predicha] * 100
    categoria_predicha = categorias[clase_predicha]
    
    # VALIDACIÓN MULTI-NIVEL
    es_valido = True
    mensaje_validacion = ""
    tipo_alerta = "success"
    
    # Nivel 1: Verificar si es clasificado como No-Cacao
    if categoria_predicha == 'No-Cacao':
        es_valido = False
        mensaje_validacion = "⚠️  Esta imagen NO parece ser una hoja de cacao"
        tipo_alerta = "no-cacao"
    
    # Nivel 2: Verificar umbral de confianza (incluso para No-Cacao)
    elif confianza < umbral_confianza * 100:
        es_valido = False
        mensaje_validacion = f"⚠️  Confianza baja ({confianza:.1f}%). Imagen ambigua o no es hoja de cacao"
        tipo_alerta = "baja-confianza"
    
    # Nivel 3: Verificar distribución de probabilidades
    # Si varias clases tienen probabilidades similares, es sospechoso
    prob_ordenadas = np.sort(predicciones[0])[::-1]
    if prob_ordenadas[0] - prob_ordenadas[1] < 0.15:  # Diferencia menor a 15%
        mensaje_validacion += "\n   ⚠️  Las probabilidades están muy dispersas. Imagen poco clara."
        if es_valido:
            tipo_alerta = "warning"
    
    # Preparar resultado
    resultado = {
        'es_valido': es_valido,
        'categoria': categoria_predicha,
        'confianza': confianza,
        'probabilidades': {cat: float(predicciones[0][i]*100) for i, cat in enumerate(categorias)},
        'mensaje': mensaje_validacion,
        'tipo_alerta': tipo_alerta
    }
    
    if mostrar_grafica:
        # Visualización
        fig = plt.figure(figsize=(14, 6))
        
        # Imagen
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        
        # Color del título según validación
        if es_valido:
            color_titulo = 'green'
            titulo = f'✅ VÁLIDA: {categoria_predicha}'
        elif tipo_alerta == "no-cacao":
            color_titulo = 'red'
            titulo = '❌ NO ES HOJA DE CACAO'
        else:
            color_titulo = 'orange'
            titulo = f'⚠️  DUDOSA: {categoria_predicha}'
        
        plt.title(titulo, fontsize=13, fontweight='bold', color=color_titulo, pad=10)
        
        # Gráfico de barras
        ax2 = plt.subplot(1, 2, 2)
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95A5A6']
        y_pos = np.arange(len(categorias))
        
        barras = plt.barh(y_pos, predicciones[0] * 100, color=colores)
        
        # Resaltar la clase predicha
        barras[clase_predicha].set_color('#2ECC71' if es_valido else '#E74C3C')
        barras[clase_predicha].set_edgecolor('black')
        barras[clase_predicha].set_linewidth(2)
        
        plt.yticks(y_pos, categorias)
        plt.xlabel('Probabilidad (%)', fontsize=11, fontweight='bold')
        plt.title(f'Distribución de Probabilidades', fontsize=12, fontweight='bold')
        plt.xlim(0, 100)
        
        # Añadir valores en las barras
        for i, v in enumerate(predicciones[0] * 100):
            plt.text(v + 2, i, f'{v:.1f}%', va='center', fontsize=10, 
                    fontweight='bold' if i == clase_predicha else 'normal')
        
        # Línea del umbral
        plt.axvline(x=umbral_confianza*100, color='red', linestyle='--', 
                   linewidth=1.5, label=f'Umbral mínimo ({umbral_confianza*100}%)')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.2, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    # Imprimir resultados
    print("\n" + "="*60)
    if es_valido:
        print("✅ RESULTADO: IMAGEN VÁLIDA")
        print(f"   🌿 Deficiencia detectada: {categoria_predicha}")
        print(f"   📊 Confianza: {confianza:.2f}%")
    else:
        print("⚠️  ALERTA: IMAGEN NO VÁLIDA O DUDOSA")
        print(mensaje_validacion)
        if tipo_alerta != "no-cacao":
            print(f"   Clasificación tentativa: {categoria_predicha}")
            print(f"   Confianza: {confianza:.2f}%")
    
    print("\n📊 Probabilidades detalladas:")
    for cat, prob in resultado['probabilidades'].items():
        emoji = "🌿" if cat in ['Potasio', 'Nitrogeno', 'Fosforo'] else "❌"
        marcador = " ← PREDICCIÓN" if cat == categoria_predicha else ""
        print(f"   {emoji} {cat}: {prob:.2f}%{marcador}")
    
    print("="*60)
    
    return resultado

# ========================================
# RESUMEN FINAL
# ========================================

print("\n" + "="*60)
print("✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
print("="*60)

print(f"\n📂 Archivos generados:")
print(f"   • Mejor modelo: {mejor_modelo_path}")
print(f"   • Modelo final: {modelo_final_path}")
print(f"   • Gráficas: {resultados_img_path}")

print("\n📝 CÓMO USAR EL MODELO:")
print("\n1. Predecir con validación automática:")
print("   resultado = predecir_deficiencia(r'ruta\\imagen.jpg')")

print("\n2. Ajustar umbral de confianza:")
print("   predecir_deficiencia(r'imagen.jpg', umbral_confianza=0.70)")

print("\n3. Predicción sin gráficas:")
print("   resultado = predecir_deficiencia(r'imagen.jpg', mostrar_grafica=False)")

print("\n4. Acceder a los resultados:")
print("   if resultado['es_valido']:")
print("       print(f\"Deficiencia: {resultado['categoria']}\")")

print("\n💡 CONSEJOS:")
print("   • Añade mínimo 50-100 imágenes variadas a la carpeta 'No-Cacao'")
print("   • Incluye imágenes de: otras plantas, objetos, fondos, etc.")
print("   • Ajusta CONFIANZA_MINIMA según tu tolerancia a falsos positivos")
print("   • Un umbral alto (0.80) = más estricto, menos falsos positivos")
print("   • Un umbral bajo (0.50) = más permisivo, detecta más casos dudosos")

print("\n" + "="*60)

# ========================================
# PRUEBA AUTOMATIZADA
# ========================================

print("\n🧪 Realizando pruebas automatizadas...\n")

try:
    # Probar con varias imágenes del validation set
    x_val, y_val = next(validation_generator)
    
    correctas = 0
    total_pruebas = min(5, len(x_val))
    
    for i in range(total_pruebas):
        img_test = x_val[i:i+1]
        pred = model.predict(img_test, verbose=0)
        clase_real = np.argmax(y_val[i])
        clase_pred = np.argmax(pred[0])
        confianza_pred = pred[0][clase_pred] * 100
        
        es_correcta = clase_real == clase_pred
        if es_correcta:
            correctas += 1
        
        # Mostrar resultado
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_test[0])
        ax.axis('off')
        
        titulo = f'Real: {categorias[clase_real]} | Pred: {categorias[clase_pred]}\n'
        titulo += f'Confianza: {confianza_pred:.1f}%'
        
        if es_correcta:
            titulo = '✅ ' + titulo
            color = 'green'
        else:
            titulo = '❌ ' + titulo
            color = 'red'
        
        ax.set_title(titulo, fontsize=11, fontweight='bold', color=color, pad=10)
        plt.tight_layout()
        plt.show()
    
    print(f"\n Resultados de prueba: {correctas}/{total_pruebas} correctas ({correctas/total_pruebas*100:.1f}%)")
    print("\n ¡Modelo con validación listo para usar!")
    
except Exception as e:
    print(f" No se pudo realizar la prueba automática: {e}")
    print("   Pero el modelo está entrenado y guardado correctamente.")

print("\n" + "="*60)