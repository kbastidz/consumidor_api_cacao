@echo off
REM ========================================
REM Script para Iniciar API Automáticamente
REM ========================================

echo.
echo ============================================================
echo  INICIO RAPIDO - DETECTOR DE DEFICIENCIAS EN CACAO
echo ============================================================
echo.

cd /d C:\Users\User\Desktop\api

REM Verificar main.py
if not exist "main.py" (
    echo ERROR: No se encontró main.py
    pause
    exit /b 1
)

echo [1/1] Iniciando API con main.py...
echo.
echo ============================================================
echo  API INICIADA
echo ============================================================
echo.
echo  URL Base:          http://localhost:8000
echo  Documentacion:     http://localhost:8000/docs
echo  Health Check:      http://localhost:8000/health
echo.
echo  Presiona CTRL+C para detener el servidor
echo ============================================================
echo.

python main.py

REM Si la API se cierra con error
if errorlevel 1 (
    echo.
    echo ERROR: La API se cerro inesperadamente
    pause
)

pause