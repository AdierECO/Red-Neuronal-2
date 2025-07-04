<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pet Classifier - Clasificador de Perros/Gatos</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            margin-top: 25px;
        }
        code {
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .note {
            background: #e7f5fe;
            border-left: 4px solid #3498db;
            padding: 10px;
            margin: 15px 0;
        }
        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 15px 0;
        }
        .terminal {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 10px 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <h1>🐶🐱 Pet Classifier</h1>
    <p>Clasificador de imágenes perros/gatos usando TensorFlow/Keras y Flask</p>

    <h2>📋 Requisitos</h2>
    <ul>
        <li>Python 3.8+</li>
        <li>TensorFlow 2.x</li>
        <li>Flask</li>
        <li>Flask-SocketIO</li>
    </ul>

    <h2>🚀 Instalación</h2>
    <div class="terminal">
        # Clonar el repositorio<br>
        git clone https://github.com/tu-usuario/pet_classifier.git<br>
        cd pet_classifier<br><br>
        
        # Crear entorno virtual (recomendado)<br>
        python -m venv venv<br>
        source venv/bin/activate  # Linux/Mac<br>
        .\venv\Scripts\activate  # Windows<br><br>
        
        # Instalar dependencias<br>
        pip install -r requirements.txt
    </div>

    <h2>⚙️ Configuración</h2>
    <p>Crea la estructura de directorios necesaria:</p>
    <pre>data/
├── train/
│   ├── dogs/
│   └── cats/
└── validation/
    ├── dogs/
    └── cats/</pre>

    <div class="note">
        <strong>Nota:</strong> Las carpetas ya incluyen archivos <code>.gitkeep</code> para mantener la estructura.
    </div>

    <h2>🖼️ Subir imágenes de entrenamiento</h2>
    <ol>
        <li>Accede a <code>/upload-train</code> en la aplicación</li>
        <li>Selecciona imágenes de perros y gatos</li>
        <li>El sistema las dividirá automáticamente en entrenamiento/validación</li>
    </ol>

    <h2>🏋️ Entrenar el modelo</h2>
    <div class="terminal">
        # Ejecutar la aplicación<br>
        python app.py<br><br>
        
        # Luego acceder a:<br>
        http://localhost:5000/train<br><br>
        
        # Hacer clic en "Iniciar Entrenamiento"
    </div>

    <div class="warning">
        <strong>Requisitos mínimos:</strong> Se recomienda GPU para entrenamiento. Con CPU puede ser muy lento.
    </div>

    <h2>🔍 Realizar predicciones</h2>
    <ol>
        <li>Sube una imagen desde la página principal (<code>/</code>)</li>
        <li>El modelo devolverá si es perro o gato con su confianza</li>
    </ol>

    <h2>📊 Estructura del Proyecto</h2>
    <pre>pet_classifier/
├── app.py                # Aplicación principal Flask
├── train.py              # Lógica de entrenamiento
├── requirements.txt      # Dependencias
├── static/               # Archivos estáticos
│   └── uploads/          # Imágenes para predicción
├── templates/            # Vistas HTML
├── data/                 # Dataset de entrenamiento
└── models/               # Modelos entrenados</pre>

    <h2>📌 Notas Importantes</h2>
    <ul>
        <li>El modelo usa transfer learning con MobileNetV2</li>
        <li>Incluye fine-tuning automático</li>
        <li>Interfaz web con actualización en tiempo real del entrenamiento</li>
    </ul>

    <h2>📄 Licencia</h2>
    <p>MIT License - Libre para uso y modificación</p>
</body>
</html>
