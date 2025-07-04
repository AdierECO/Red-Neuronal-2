import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras import saving

class TrainingCallback(Callback):
    def __init__(self, socketio=None):
        self.socketio = socketio
    
    def on_epoch_end(self, epoch, logs=None):
        if self.socketio:
            progress = (epoch + 1) / self.params['epochs'] * 100
            self.socketio.emit('training_update', {
                'epoch': epoch + 1,
                'total_epochs': self.params['epochs'],
                'loss': logs['loss'],
                'accuracy': logs['accuracy'],
                'val_loss': logs['val_loss'],
                'val_accuracy': logs['val_accuracy'],
                'progress': progress
            })

def train_model(train_dir, val_dir, socketio=None, epochs=20):
    # Configuración
    img_size = (224, 224)
    batch_size = 32
    
    # Generadores de datos
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    # Modelo base
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar capas base inicialmente
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callback de Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=1,
        restore_best_weights=True
    )
    
    # Entrenamiento inicial
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[TrainingCallback(socketio), early_stopping]
    )
    
    # Obtener la mejor época de manera segura
    if hasattr(early_stopping, 'best_epoch'):
        best_epoch = early_stopping.best_epoch
    else:
        best_epoch = len(history.history['accuracy']) - 1  # Última época si no hubo early stopping
    
    # Asegurar que el índice esté dentro del rango válido
    best_epoch = min(best_epoch, len(history.history['accuracy']) - 1)
    best_epoch = max(best_epoch, 0)  # Asegurar que no sea negativo
    
    # Fine-tuning: Descongelar algunas capas
    base_model.trainable = True
    fine_tune_at = 100
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Continuar entrenamiento con fine-tuning
    history_fine = model.fit(
        train_generator,
        epochs=epochs,
        initial_epoch=history.epoch[-1] + 1,
        validation_data=val_generator,
        callbacks=[TrainingCallback(socketio), early_stopping]
    )
    
    # Determinar la mejor época considerando fine-tuning
    if hasattr(early_stopping, 'best_epoch'):
        best_epoch_fine = early_stopping.best_epoch
    else:
        best_epoch_fine = len(history_fine.history['accuracy']) - 1
    
    best_epoch_fine = min(best_epoch_fine, len(history_fine.history['accuracy']) - 1)
    best_epoch_fine = max(best_epoch_fine, 0)
    
    # Combinar historiales
    full_history = {
        'accuracy': history.history['accuracy'] + history_fine.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'] + history_fine.history['val_accuracy'],
        'loss': history.history['loss'] + history_fine.history['loss'],
        'val_loss': history.history['val_loss'] + history_fine.history['val_loss']
    }
    
    # La mejor época global es la mejor del fine-tuning ajustada por el offset
    global_best_epoch = len(history.history['accuracy']) + best_epoch_fine
    
    # Métricas finales (usando los mejores pesos)
    metrics = {
        'train_accuracy': full_history['accuracy'][global_best_epoch],
        'val_accuracy': full_history['val_accuracy'][global_best_epoch],
        'train_loss': full_history['loss'][global_best_epoch],
        'val_loss': full_history['val_loss'][global_best_epoch],
        'epochs_used': global_best_epoch + 1,
        'early_stopped': hasattr(early_stopping, 'stopped_epoch'),
        'best_epoch': global_best_epoch + 1,
        'history': full_history
    }
    
    # Guardar modelo en formato .keras
    os.makedirs('models', exist_ok=True)
    model_path = 'models/pet_classifier.keras'
    saving.save_model(model, model_path)
    
    return {
        'model_path': model_path,
        'metrics': metrics
    }