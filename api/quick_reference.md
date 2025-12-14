# 🚀 Guía Rápida de Referencia - API Cacao

## 📋 Comandos Esenciales

### Iniciar API
```bash
python api.py
```

### Verificar Estado
```bash
curl http://localhost:8000/health
```

### Predecir Imagen
```bash
curl -X POST http://localhost:8000/predict -F "file=@imagen.jpg"
```

---

## 🔗 URLs Importantes

| URL | Descripción |
|-----|-------------|
| `http://localhost:8000` | API Base |
| `http://localhost:8000/docs` | Documentación Interactiva |
| `http://localhost:8000/health` | Estado del Servicio |
| `index.html` | Frontend Web |

---

## 📡 API Endpoints

### 1. Health Check
```http
GET /health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "modelo_cargado": true,
  "categorias": ["Potasio", "Nitrogeno", "Fosforo"]
}
```

### 2. Predecir
```http
POST /predict
Content-Type: multipart/form-data

file: [imagen.jpg]
```

**Respuesta:**
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
  "archivo": "imagen.jpg"
}
```

---

## 💻 Ejemplos de Código

### JavaScript (Fetch)
```javascript
const formData = new FormData();
formData.append('file', archivo);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

### JavaScript (Axios)
```javascript
const formData = new FormData();
formData.append('file', archivo);

axios.post('http://localhost:8000/predict', formData)
  .then(res => console.log(res.data));
```

### Python (requests)
```python
import requests

files = {'file': open('imagen.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@imagen.jpg"
```

---

## 🔧 Configuración Rápida

### Cambiar Puerto
```python
# En api.py, última línea:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Cambiar Ruta del Modelo
```python
# En api.py, línea 19:
MODEL_PATH = r'C:\nueva\ruta\modelo.h5'
```

### Limitar Orígenes CORS
```python
# En api.py, agregar después de crear app:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://tu-dominio.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🐛 Errores Comunes

### Error 1: Puerto ocupado
```bash
# Verificar puerto
netstat -ano | findstr :8000

# Matar proceso (Windows)
taskkill /PID [numero_pid] /F

# Cambiar puerto en api.py
```

### Error 2: Modelo no encontrado
```
FileNotFoundError: No se encuentra el modelo
```
**Solución:** Verificar ruta en `MODEL_PATH`

### Error 3: CORS bloqueado
```
Access to fetch blocked by CORS policy
```
**Solución:** API ya tiene CORS habilitado. Verificar URL correcta.

### Error 4: Módulo no encontrado
```bash
pip install fastapi uvicorn python-multipart
```

---

## 📊 Códigos de Estado HTTP

| Código | Significado |
|--------|-------------|
| 200 | ✅ Éxito |
| 400 | ❌ Solicitud incorrecta |
| 404 | ❌ Endpoint no encontrado |
| 500 | ❌ Error del servidor |

---

## 🎯 Testing Rápido

### Con Postman
1. Método: POST
2. URL: `http://localhost:8000/predict`
3. Body → form-data
4. Key: `file` (tipo File)
5. Value: Seleccionar imagen
6. Send

### Con Frontend
1. Abrir `index.html`
2. Arrastrar imagen
3. Click "Analizar"

### Con Python Script
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    files={'file': open('test.jpg', 'rb')}
)
print(response.json())
```

---

## 📁 Archivos Necesarios

```
✅ api.py                    (API)
✅ modelo_final_cacao.h5     (Modelo)
✅ index.html                (Frontend - opcional)
```

---

## 🔍 Verificación del Sistema

```bash
# 1. Verificar Python
python --version

# 2. Verificar dependencias
pip list | findstr "tensorflow fastapi uvicorn"

# 3. Verificar modelo existe
dir modelo_final_cacao.h5

# 4. Iniciar API
python api.py

# 5. Probar en otra terminal
curl http://localhost:8000/health
```

---

## 💡 Tips Útiles

1. **Primera predicción es lenta** → Normal (carga el modelo)
2. **Usar imágenes claras** → Mejor precisión
3. **Formato recomendado** → JPG 224x224 píxeles
4. **Ver logs en consola** → Útil para debug
5. **Usar /docs** → Documentación interactiva de FastAPI

---

## 📞 Soporte

Si tienes problemas:

1. ✅ Verificar API ejecutándose: `curl http://localhost:8000/health`
2. ✅ Verificar modelo existe en ruta correcta
3. ✅ Revisar logs en consola donde ejecutas `python api.py`
4. ✅ Probar con `/docs` para interfaz de pruebas

---

## 🎓 Recursos

- [Documentación FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow Docs](https://www.tensorflow.org/)
- [Documentación Completa](./DOCUMENTACION_COMPLETA.md)

---

*Última actualización: Diciembre 2024*