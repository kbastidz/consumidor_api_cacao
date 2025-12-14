@echo off
REM ========================================
REM Script para Entrenar Modelo de Cacao
REM ========================================

echo.
echo ============================================================
echo  ENTRENAMIENTO DE MODELO - DETECTOR DE DEFICIENCIAS CACAO
echo ============================================================
echo.

REM Cambiar al directorio del modelo
cd /d C:\Users\User\Desktop\local

echo [1/3] Verificando directorio...
if not exist "train_model.py" (
    echo ERROR: No se encuentra train_model.py
    echo Verifica que estas en la carpeta correcta
    pause
    exit /b 1
)
echo OK - Directorio correcto
echo.

echo [2/3] Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Fallo la instalacion de dependencias
    pause
    exit /b 1
)
echo OK - Dependencias instaladas
echo.

echo [3/3] Iniciando entrenamiento...
echo NOTA: Esto puede tardar varios minutos dependiendo de tu hardware
echo.
python train_model.py

if errorlevel 1 (
    echo.
    echo ERROR: El entrenamiento fallo
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  ENTRENAMIENTO COMPLETADO EXITOSAMENTE
echo ============================================================
echo.
echo Archivos generados:
echo  - modelo_final_cacao.h5
echo  - mejor_modelo_cacao.h5
echo  - resultados_entrenamiento_cacao.png
echo.

pause