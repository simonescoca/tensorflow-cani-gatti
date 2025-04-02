import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import platform
import os
import sys

# Verifica ambiente
print("\n=== Configurazione Sistema ===")
print(f"TensorFlow {tf.__version__}")
print(f"Python {platform.python_version()}")
print(f"GPU disponibili: {tf.config.list_physical_devices('GPU')}")
print(f"OS: {platform.system()} {platform.mac_ver()[0]}\n")

# 1. Preparazione dati
print("=== Preparazione Dataset ===")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
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

print(f"\nClassi riconosciute: {train_generator.class_indices}")
print(f"Immagini training: {train_generator.samples}")
print(f"Immagini validation: {val_generator.samples}\n")

# 2. Definizione modello (ottimizzato per 150x150)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),  # Ridotto per 150x150
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 3. Addestramento
print("\n=== Avvio Addestramento ===")
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=3, monitor='val_loss'),
        TensorBoard(log_dir='./logs')
    ],
    verbose=1
)

# 4. Salvataggio
model.save('modello_animali_150x150.h5')
print("\n✅ Modello salvato come 'modello_animali_150x150.h5'")