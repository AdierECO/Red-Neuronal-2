import os
import random
import time
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
socketio = SocketIO(app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configuración de directorios
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TRAIN_FOLDER'] = 'data/train'
app.config['VAL_FOLDER'] = 'data/validation'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Crear directorios necesarios
for folder in [app.config['UPLOAD_FOLDER'], app.config['TRAIN_FOLDER'], app.config['VAL_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
    for category in ['dogs', 'cats']:
        os.makedirs(os.path.join(folder, category), exist_ok=True)

# --- Funciones auxiliares ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def count_images(folder_path):
    try:
        return {
            'dogs': len(os.listdir(os.path.join(folder_path, 'dogs'))),
            'cats': len(os.listdir(os.path.join(folder_path, 'cats')))
        }
    except FileNotFoundError:
        return {'dogs': 0, 'cats': 0}

# Función faltante añadida
def get_total_counts():
    train_counts = count_images(app.config['TRAIN_FOLDER'])
    val_counts = count_images(app.config['VAL_FOLDER'])
    return {
        'dogs': train_counts['dogs'] + val_counts['dogs'],
        'cats': train_counts['cats'] + val_counts['cats']
    }

# --- Rutas principales ---
@app.route('/')
def index():
    model_exists = os.path.exists('models/pet_classifier.keras')
    return render_template('index.html', model_loaded=model_exists)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Cargar el modelo
            model_path = 'models/pet_classifier.keras'
            if not os.path.exists(model_path):
                return jsonify({'error': 'Modelo no encontrado. Entrena el modelo primero.'}), 400
                
            model = tf.keras.models.load_model(model_path)
            
            # Preprocesar la imagen
            img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = img_array / 255.0
            
            # Realizar predicción
            predictions = model.predict(img_array)
            score = float(predictions[0][0])
            
            label = "Perro" if score > 0.5 else "Gato"
            confidence = round(max(score, 1 - score) * 100, 2)
            
            # Asegurar que la imagen esté disponible
            if not os.path.exists(filepath):
                return jsonify({'error': 'Error al guardar la imagen'}), 500
            
            return jsonify({
                'prediction': label,
                'confidence': confidence,
                'image_path': f"/{filepath}"  # Añadir / para ruta absoluta
            })
            
        except Exception as e:
            return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500
    
    return jsonify({'error': 'Formato de archivo no permitido'}), 400

# --- Rutas de entrenamiento ---
@app.route('/train')
def train_page():
    train_counts = count_images(app.config['TRAIN_FOLDER'])
    val_counts = count_images(app.config['VAL_FOLDER'])
    return render_template('train.html', 
                         train_counts=train_counts,
                         val_counts=val_counts)

@app.route('/start-training', methods=['POST'])
def start_training():
    try:
        socketio.emit('training_status', {
            'message': 'Iniciando entrenamiento...',
            'progress': 0,
            'early_stopping': False
        })
        
        def training_task():
            try:
                result = train_model(
                    train_dir=app.config['TRAIN_FOLDER'],
                    val_dir=app.config['VAL_FOLDER'],
                    socketio=socketio
                )
                
                socketio.emit('training_complete', {
                    'message': '¡Entrenamiento completado!',
                    'metrics': result['metrics'],
                    'history': result['metrics']['history'],
                    'early_stopped': result['metrics']['early_stopped'],
                    'best_epoch': result['metrics']['best_epoch']
                })
            except Exception as e:
                socketio.emit('training_error', {
                    'message': f'Error: {str(e)}',
                    'early_stopping': False
                })
        
        socketio.start_background_task(training_task)
        return {'status': 'started'}, 200
        
    except Exception as e:
        return {'error': str(e)}, 500

# --- Rutas para subir imágenes ---
@app.route('/upload-train')
def upload_train_page():
    total_counts = get_total_counts()  # Ahora esta función está definida
    return render_template('upload_train.html', 
                         total_dogs=total_counts['dogs'],
                         total_cats=total_counts['cats'])

@app.route('/upload-training-batch', methods=['POST'])
def upload_training_batch():
    try:
        # Determinar tipo de archivo (perros o gatos)
        if 'dogs' in request.files:
            file_type = 'dogs'
        elif 'cats' in request.files:
            file_type = 'cats'
        else:
            return jsonify({'error': 'No se especificó tipo de archivo'}), 400
            
        files = request.files.getlist(file_type)
        train_ratio = float(request.form.get('train_ratio', 0.8))
        
        train_count = 0
        val_count = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                # Nombre único con timestamp y hash
                file_hash = hash(file.filename + str(time.time()))
                filename = f"{file_hash}_{secure_filename(file.filename)}"
                
                if random.random() < train_ratio:
                    save_path = os.path.join(app.config['TRAIN_FOLDER'], file_type, filename)
                    train_count += 1
                else:
                    save_path = os.path.join(app.config['VAL_FOLDER'], file_type, filename)
                    val_count += 1
                
                # Guardar en chunks
                chunk_size = 4096
                with open(save_path, 'wb') as f:
                    while True:
                        chunk = file.stream.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
        
        return jsonify({
            'message': f'Lote de {len(files)} {file_type} procesado',
            'train': train_count,
            'validation': val_count,
            'file_type': file_type
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-image-counts')
def get_image_counts():
    return jsonify(get_total_counts())  # Usamos la función definida

# --- Ruta de resultados ---
@app.route('/results')
def results_page():
    # En producción, estos datos deberían venir de una base de datos
    metrics = {
        'train_accuracy': 0.92,
        'val_accuracy': 0.88,
        'train_loss': 0.15,
        'val_loss': 0.18,
        'epochs_used': 10,
        'early_stopped': False,
        'history': {
            'accuracy': [0.65, 0.72, 0.78, 0.82, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92],
            'val_accuracy': [0.63, 0.70, 0.75, 0.80, 0.83, 0.85, 0.86, 0.87, 0.88, 0.88],
            'loss': [0.65, 0.55, 0.45, 0.38, 0.32, 0.27, 0.23, 0.20, 0.18, 0.15],
            'val_loss': [0.67, 0.57, 0.48, 0.41, 0.35, 0.30, 0.26, 0.23, 0.20, 0.18]
        }
    }
    return render_template('results.html', metrics=metrics)

# --- WebSocket Events ---
@socketio.on('connect')
def handle_connect():
    emit('training_status', {
        'message': 'Conectado al monitor de entrenamiento',
        'progress': 0,
        'early_stopping': False
    })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)