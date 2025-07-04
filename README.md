# 🐶🐱 Sistema de Clasificación de Perros y Gatos

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0.2-green" alt="Flask">
  <img src="https://img.shields.io/badge/TensorFlow-2.16.1-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/MobileNetV2-Transfer_Learning-yellow" alt="MobileNetV2">
</p>

## 📌 Descripción
Aplicación web que clasifica imágenes entre perros y gatos usando transfer learning con MobileNetV2 y una interfaz Flask con actualización en tiempo real del entrenamiento.

## 🚀 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/AdierECO/pet_classifier.git
cd pet_classifier

# 2. Crear entorno virtual (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## ⚙️ Configuración

```bash
Prepara tu dataset:

Crea la estructura de carpetas:
data/
└── train/
    ├── dogs/    # Coloca aquí imágenes de perros
    └── cats/    # Coloca aquí imágenes de gatos

Recomendado: Mínimo 100 imágenes por categoría (formato JPG/PNG)
```

## 🖥️ Uso

```bash
# Iniciar la aplicación Flask
python app.py

Accede a la interfaz web en: http://localhost:5000

Funcionalidades principales:
1. Entrenamiento: Monitoreo en tiempo real con Flask-SocketIO
2. Predicción: Sube imágenes para clasificación
3. Gestión de dataset: Carga imágenes directamente desde la web
```

## 📂 Estructura del Proyecto

```bash
pet_classifier/
├── app.py                # Aplicación principal Flask
├── train.py              # Lógica de entrenamiento con MobileNetV2
├── static/               # Archivos estáticos (uploads)
├── templates/            # Vistas HTML
├── data/                 # Dataset de entrenamiento (no incluido)
│   ├── train/            # Imágenes de entrenamiento
│   └── validation/       # Validación automática (80/20)
└── models/               # Modelos entrenados (generados automáticamente)
```

## 🏋️ Entrenamiento Avanzado

```bash
Opciones de entrenamiento:
# Entrenamiento básico (congelando capas)
python train.py --mode basic --epochs 20

# Fine-tuning (descongelando capas superiores)
python train.py --mode fine_tune --epochs 10

Parámetros configurables:
--batch_size: Tamaño del lote (default: 32)
--learning_rate: Tasa de aprendizaje (default: 0.001)
```

## ⚠️ Notas Importantes

```bash
1. GPU altamente recomendada para entrenamiento (5x más rápido)
2. El modelo usa Data Augmentation automático
3. Incluye Early Stopping para evitar overfitting
4. Los modelos se guardan en formato .keras (optimizados para producción)
```

## 📊 Métricas Esperadas

| Métrica         | Entrenamiento | Validación |
|-----------------|---------------|------------|
| Precisión       | 92-96%        | 88-92%     |
| Pérdida         | 0.10-0.15     | 0.15-0.20  |
| Tiempo (GPU)    | ~5 min        |            |
| Tiempo (CPU)    | ~30 min       |            |

## 📄 Licencia
MIT License - Libre para uso académico y comercial.

<div align="center">
  <p>✉️ <strong>Contacto</strong>: adierortix@gmail.com.com | 🌐 <a href="https://github.com/AdierECO">GitHub</a></p>
</div>
