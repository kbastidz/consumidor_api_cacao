"""
Script de Diagnóstico para el Modelo de Detección de Cacao
Verifica si el modelo tiene 3 o 4 clases y su estado
"""

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from PIL import Image

print("="*70)
print("🔍 DIAGNÓSTICO DEL MODELO DE DETECCIÓN DE CACAO")
print("="*70)

# ========================================
# CONFIGURACIÓN
# ========================================

BASE_PATH = r'C:\Users\User\Desktop\local'
DATASET_PATH = os.path.join(BASE_PATH, 'dataset_cacao')

# Modelos a verificar
MODELO_ANTIGUO = os.path.join(BASE_PATH, 'modelo_final_cacao.h5')
MODELO_NUEVO = os.path.join(BASE_PATH, 'modelo_final_cacao_validado.h5')

print("\n📂 Verificando archivos...")

# ========================================
# 1. VERIFICAR MODELOS EXISTENTES
# ========================================

print("\n1️⃣ MODELOS GUARDADOS:")
print("-" * 70)

if os.path.exists(MODELO_ANTIGUO):
    size = os.path.getsize(MODELO_ANTIGUO) / (1024*1024)
    print(f"   ✅ Modelo antiguo (3 clases): {MODELO_ANTIGUO}")
    print(f"      Tamaño: {size:.2f} MB")
else:
    print(f"   ❌ Modelo antiguo NO encontrado: {MODELO_ANTIGUO}")

if os.path.exists(MODELO_NUEVO):
    size = os.path.getsize(MODELO_NUEVO) / (1024*1024)
    print(f"   ✅ Modelo nuevo (4 clases): {MODELO_NUEVO}")
    print(f"      Tamaño: {size:.2f} MB")
else:
    print(f"   ❌ Modelo nuevo NO encontrado: {MODELO_NUEVO}")
    print(f"      ⚠️  ESTE ES EL PROBLEMA: Necesitas entrenar el modelo nuevo")

# ========================================
# 2. VERIFICAR DATASET
# ========================================

print("\n2️⃣ ESTRUCTURA DEL DATASET:")
print("-" * 70)

categorias_esperadas = ['Potasio', 'Nitrogeno', 'Fosforo', 'No-Cacao']

if not os.path.exists(DATASET_PATH):
    print(f"   ❌ Dataset NO encontrado: {DATASET_PATH}")
else:
    print(f"   📁 Dataset encontrado: {DATASET_PATH}\n")
    
    total_imagenes = 0
    problemas = []
    
    for categoria in categorias_esperadas:
        path = os.path.join(DATASET_PATH, categoria)
        
        if os.path.exists(path):
            imagenes = [f for f in os.listdir(path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            num_imgs = len(imagenes)
            total_imagenes += num_imgs
            
            # Verificar si hay suficientes imágenes
            status = "✅"
            nota = ""
            
            if categoria == 'No-Cacao':
                if num_imgs == 0:
                    status = "❌"
                    nota = "← ¡PROBLEMA! Carpeta vacía"
                    problemas.append(f"La carpeta '{categoria}' está vacía")
                elif num_imgs < 30:
                    status = "⚠️ "
                    nota = f"← Pocas imágenes (recomendado: 50+)"
                    problemas.append(f"La carpeta '{categoria}' tiene solo {num_imgs} imágenes (mínimo recomendado: 50)")
            else:
                if num_imgs == 0:
                    status = "❌"
                    nota = "← Carpeta vacía"
                    problemas.append(f"La carpeta '{categoria}' está vacía")
                elif num_imgs < 20:
                    status = "⚠️ "
                    nota = f"← Pocas imágenes"
            
            print(f"   {status} {categoria:15} {num_imgs:4} imágenes   {nota}")
        else:
            print(f"   ❌ {categoria:15} NO EXISTE")
            problemas.append(f"Falta crear la carpeta '{categoria}'")
    
    print(f"\n   📊 Total: {total_imagenes} imágenes")
    
    if problemas:
        print(f"\n   ⚠️  PROBLEMAS DETECTADOS:")
        for i, prob in enumerate(problemas, 1):
            print(f"      {i}. {prob}")

# ========================================
# 3. INSPECCIONAR MODELO ACTUAL
# ========================================

print("\n3️⃣ ANÁLISIS DEL MODELO:")
print("-" * 70)

# Intentar cargar el modelo que está usando la API
modelo_api = MODELO_NUEVO if os.path.exists(MODELO_NUEVO) else MODELO_ANTIGUO

if os.path.exists(modelo_api):
    try:
        print(f"   Cargando: {os.path.basename(modelo_api)}...")
        model = keras.models.load_model(modelo_api)
        
        # Obtener información del modelo
        output_shape = model.output_shape
        num_clases = output_shape[-1]
        
        print(f"\n   ✅ Modelo cargado exitosamente")
        print(f"   📊 Número de clases: {num_clases}")
        
        if num_clases == 3:
            print(f"   ⚠️  ESTE ES UN MODELO ANTIGUO (3 clases)")
            print(f"      Clases: Potasio, Nitrógeno, Fósforo")
            print(f"      ❌ NO tiene validación de No-Cacao")
            print(f"\n   🔧 SOLUCIÓN: Entrena el modelo nuevo con 4 clases")
        elif num_clases == 4:
            print(f"   ✅ ESTE ES UN MODELO NUEVO (4 clases)")
            print(f"      Clases: Potasio, Nitrógeno, Fósforo, No-Cacao")
            print(f"      ✅ Tiene validación de No-Cacao")
        else:
            print(f"   ❓ Número de clases inesperado: {num_clases}")
        
        # Mostrar arquitectura resumida
        print(f"\n   🏗️  Arquitectura:")
        print(f"      • Parámetros totales: {model.count_params():,}")
        print(f"      • Capas: {len(model.layers)}")
        print(f"      • Input shape: {model.input_shape}")
        print(f"      • Output shape: {model.output_shape}")
        
        # Probar con una imagen de prueba (si existe)
        print(f"\n   🧪 Probando predicción...")
        
        # Crear imagen de prueba (ruido aleatorio)
        img_test = np.random.rand(1, 224, 224, 3)
        pred = model.predict(img_test, verbose=0)
        
        print(f"      • Shape de predicción: {pred.shape}")
        print(f"      • Suma de probabilidades: {pred[0].sum():.4f}")
        
        if num_clases == 3:
            categorias_test = ['Potasio', 'Nitrogeno', 'Fosforo']
        else:
            categorias_test = ['Potasio', 'Nitrogeno', 'Fosforo', 'No-Cacao']
        
        print(f"\n      Probabilidades de prueba:")
        for i, cat in enumerate(categorias_test):
            print(f"         {cat}: {pred[0][i]*100:.2f}%")
        
    except Exception as e:
        print(f"   ❌ Error al cargar el modelo: {e}")
else:
    print(f"   ❌ No se encontró ningún modelo")

# ========================================
# 4. DIAGNÓSTICO FINAL Y RECOMENDACIONES
# ========================================

print("\n4️⃣ DIAGNÓSTICO FINAL:")
print("="*70)

# Determinar el problema
modelo_correcto_existe = os.path.exists(MODELO_NUEVO)
carpeta_no_cacao_existe = os.path.exists(os.path.join(DATASET_PATH, 'No-Cacao'))
carpeta_no_cacao_tiene_imagenes = False

if carpeta_no_cacao_existe:
    path_no_cacao = os.path.join(DATASET_PATH, 'No-Cacao')
    imagenes_no_cacao = [f for f in os.listdir(path_no_cacao) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    carpeta_no_cacao_tiene_imagenes = len(imagenes_no_cacao) >= 30

# Identificar el problema
if not modelo_correcto_existe:
    print("❌ PROBLEMA PRINCIPAL: El modelo con validación NO existe")
    print("\n📋 PASOS PARA SOLUCIONAR:")
    print("\n   PASO 1: Preparar dataset No-Cacao")
    if not carpeta_no_cacao_existe:
        print(f"      1.1 Crear carpeta: {os.path.join(DATASET_PATH, 'No-Cacao')}")
    if not carpeta_no_cacao_tiene_imagenes:
        print(f"      1.2 Agregar mínimo 50-100 imágenes variadas:")
        print(f"          • Otras plantas (flores, césped, árboles)")
        print(f"          • Animales (perros, gatos, aves, ARDILLAS)")
        print(f"          • Objetos (herramientas, manos, tierra)")
        print(f"          • Fondos (cielo, suelo, texturas)")
    
    print("\n   PASO 2: Entrenar el modelo nuevo")
    print("      python modelo_entrenamiento_validado.py")
    
    print("\n   PASO 3: Actualizar la API")
    print("      Verificar que MODEL_PATH apunte a:")
    print(f"      {MODELO_NUEVO}")

elif not carpeta_no_cacao_tiene_imagenes:
    print("⚠️  PROBLEMA: Carpeta No-Cacao sin suficientes imágenes")
    print("\n📋 SOLUCIÓN:")
    print("   1. Agregar mínimo 50-100 imágenes variadas a:")
    print(f"      {os.path.join(DATASET_PATH, 'No-Cacao')}")
    print("   2. Re-entrenar el modelo:")
    print("      python modelo_entrenamiento_validado.py")

else:
    print("✅ Todo parece estar en orden")
    print("\n   Verifica que tu API esté usando el modelo correcto:")
    print(f"   MODEL_PATH = r'{MODELO_NUEVO}'")

# ========================================
# 5. GENERAR SCRIPT DE DESCARGA DE IMÁGENES NO-CACAO
# ========================================

print("\n5️⃣ AYUDA EXTRA: Obtener imágenes No-Cacao")
print("-" * 70)
print("""
💡 OPCIÓN 1: Descargar desde internet
   Busca y descarga imágenes de:
   - Unsplash.com (búsqueda: "leaves", "flowers", "animals")
   - Pixabay.com (búsqueda: "nature", "objects")
   - Pexels.com (búsqueda: "random objects")

💡 OPCIÓN 2: Usar tus propias fotos
   Toma fotos de tu entorno:
   - Otras plantas de tu jardín
   - Objetos cotidianos
   - Mascotas
   - Texturas variadas

💡 OPCIÓN 3: Dataset público
   Usa un subset de ImageNet o COCO dataset
""")

print("\n" + "="*70)
print("✅ Diagnóstico completado")
print("="*70)

# ========================================
# 6. CREAR REPORTE
# ========================================

reporte_path = os.path.join(BASE_PATH, 'diagnostico_modelo.txt')
with open(reporte_path, 'w', encoding='utf-8') as f:
    f.write("REPORTE DE DIAGNÓSTICO - MODELO CACAO\n")
    f.write("="*70 + "\n\n")
    f.write(f"Fecha: {np.datetime64('now')}\n\n")
    f.write(f"Modelo antiguo existe: {os.path.exists(MODELO_ANTIGUO)}\n")
    f.write(f"Modelo nuevo existe: {os.path.exists(MODELO_NUEVO)}\n")
    f.write(f"Carpeta No-Cacao existe: {carpeta_no_cacao_existe}\n")
    f.write(f"Carpeta No-Cacao tiene imágenes suficientes: {carpeta_no_cacao_tiene_imagenes}\n")

print(f"\n📄 Reporte guardado en: {reporte_path}")