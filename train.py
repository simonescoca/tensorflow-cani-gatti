import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
import platform
import os

# Configurazione ambiente
print("\n=== Configurazione Sistema ===")
print(f"TensorFlow: {tf.__version__}")
print(f"Python: {platform.python_version()}")
print(f"GPU disponibili: {tf.config.list_physical_devices('GPU')}")
print(f"OS: {platform.system()} {platform.mac_ver()[0]}\n")

# Preparazione del dataset
print("=== Preparazione Dataset ===")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset/train/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Informazioni sul dataset
print(f"\nClassi riconosciute: {train_generator.class_indices}")
print(f"Immagini di training: {train_generator.samples}")
print(f"Immagini di validazione: {val_generator.samples}\n")

# Definizione del modello con MobileNetV2
print("=== Creazione Modello (MobileNetV2) ===")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Congela i pesi pre-addestrati

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Learning rate più alto per il transfer learning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Addestramento
print("\n=== Avvio Addestramento ===")
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        TensorBoard(log_dir='./logs')
    ],
    verbose=1
)

# Salvataggio del modello
model.save('modello_animali_150x150.keras')
print("\n✅ Modello salvato come 'modello_animali_150x150.keras'")

# Risultati finali
print("\n=== Risultati Addestramento ===")
print(f"Accuratezza finale (training): {history.history['accuracy'][-1]:.2%}")
print(f"Accuratezza finale (validazione): {history.history['val_accuracy'][-1]:.2%}")