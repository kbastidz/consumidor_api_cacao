# 🌿 Detector de Deficiencias Nutricionales en Cacao

Sistema de inteligencia artificial para detectar deficiencias de **Potasio**, **Nitrógeno** y **Fósforo** en hojas de cacao mediante análisis de imágenes.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Características

- ✅ **Detección automática** de 3 tipos de deficiencias
- ✅ **API REST** lista para integración
- ✅ **Frontend web** interactivo incluido
- ✅ **Transfer Learning** con MobileNetV2
- ✅ **Precisión optimizada** con data augmentation

## 📊 Modelo

- **Arquitectura**: MobileNetV2 + Capas personalizadas
- **Input**: Imágenes 224x224 píxeles
- **Output**: Probabilidades para 3 categorías
- **Framework**: TensorFlow/Keras
- **Parámetros entrenables**: 164,355

## 🛠️ Instalación

### 1. Clonar repositorio

```bash
git clone https://github.com/tu-usuario/detector-cacao.git
cd detector-cacao
```

### 2. Instalar dependencias

```bash
# Para entrenamiento
pip install -r requirements.txt

# Para API
pip install -r requirements_api.txt
```

### 3. Entrenar modelo (opcional)

```bash
python train_model.py
```

### 4. Ejecutar API

```bash
python api.py
```

### 5. Abrir frontend

Abre `index.html` en tu navegador o visita: http://localhost:8000/docs

## 📡 Uso de la API

### Endpoint Principal

```bash
POST /predict
```

### Ejemplo con cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@hoja_cacao.jpg"
```

### Respuesta

```json
{
  "success": true,
  "data": {
    "deficiencia": "Potasio",
    "confianza": 87.45,
    "probabilidades": {
      "Potasio": 87.45,
      "Nitrogeno": 8.32,
      "Fosforo": 4.23
    }
  },
  "archivo": "hoja_cacao.jpg"
}
```

## 💻 Integración en JavaScript

```javascript
async function analizarHoja(archivo) {
  const formData = new FormData();
  formData.append('file', archivo);

  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });

  const resultado = await response.json();
  console.log(resultado.data);
}
```

## 💻 Integración en Python

```python
import requests

def predecir(imagen_path):
    url = 'http://localhost:8000/predict'
    files = {'file': open(imagen_path, 'rb')}
    response = requests.post(url, files=files)
    return response.json()

resultado = predecir('hoja.jpg')
print(resultado['data']['deficiencia'])
```

## 📂 Estructura del Proyecto

```
detector-cacao/
├── train_model.py              # Entrenamiento del modelo
├── api.py                      # API FastAPI
├── index.html                  # Frontend web
├── requirements.txt            # Dependencias entrenamiento
├── requirements_api.txt        # Dependencias API
├── modelo_final_cacao.h5       # Modelo entrenado
└── dataset_cacao/              # Dataset de imágenes
    ├── Potasio/
    ├── Nitrogeno/
    └── Fosforo/
```

## 🎯 Endpoints Disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Información de la API |
| GET | `/health` | Estado del servicio |
| GET | `/categorias` | Categorías disponibles |
| POST | `/predict` | Predecir deficiencia |

## 📖 Documentación Completa

Visita la documentación interactiva en: http://localhost:8000/docs

## 🔧 Requisitos del Sistema

- Python 3.11
- 4GB RAM mínimo
- GPU opcional (mejora velocidad)
- Windows/Linux/MacOS

## 📦 Dependencias Principales

```
tensorflow==2.16.1
fastapi==0.104.1
uvicorn==0.24.0
pillow==10.1.0
numpy==1.26.4
matplotlib==3.8.2
```

## 🐛 Solución de Problemas

### API no inicia

```bash
# Verificar puerto disponible
netstat -ano | findstr :8000

# Cambiar puerto en api.py
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Error de CORS

La API ya tiene CORS habilitado. Si persiste, especifica tu dominio en `api.py`:

```python
allow_origins=["http://tu-dominio.com"]
```

### Modelo no encontrado

Verifica la ruta en `api.py`:

```python
MODEL_PATH = r'C:\ruta\correcta\modelo_final_cacao.h5'
```

## 📈 Mejoras Futuras

- [ ] Autenticación JWT
- [ ] Base de datos de predicciones
- [ ] App móvil
- [ ] Más categorías de deficiencias
- [ ] Dashboard de estadísticas
- [ ] Exportación de reportes

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas!

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -am 'Agregar mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- Email: tu-email@ejemplo.com

## 🙏 Agradecimientos

- Dataset de hojas de cacao
- TensorFlow/Keras por el framework
- FastAPI por la excelente documentación

---

⭐ Si este proyecto te fue útil, dale una estrella en GitHub!