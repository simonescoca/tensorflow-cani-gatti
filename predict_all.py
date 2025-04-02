import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import platform
import os

# Configurazione ambiente
print("\n=== Configurazione Prediction ===")
print(f"TensorFlow: {tf.__version__}")
print(f"GPU disponibili: {tf.config.list_physical_devices('GPU')}\n")

# Caricamento del modello
try:
    model = tf.keras.models.load_model('modello_animali_150x150.keras')
    print("✅ Modello caricato correttamente")
except Exception as e:
    print(f"❌ Errore durante il caricamento del modello: {str(e)}")
    exit()

# Funzione per preparare l'immagine
def prepare_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalizzazione coerente con il training
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Errore durante l'elaborazione dell'immagine '{img_path}': {str(e)}")
        return None

# Funzione di predizione
def predict_image(img_path):
    img_array = prepare_image(img_path)
    if img_array is None:
        return  # Salta se c'è stato un errore
    
    prediction = model.predict(img_array)[0][0]  # Probabilità di "gatto"
    
    if prediction > 0.5:
        label = "Gatto"
        confidence = prediction * 100
        symbol = "🐱"
    else:
        label = "Cane"
        confidence = (1 - prediction) * 100
        symbol = "🐶"
    
    print(f"\n=== Risultato Predizione per '{img_path}' ===")
    print(f"{symbol} Predizione: {label}")
    print(f"Confidenza: {confidence:.2f}%")

# Esecuzione su tutte le immagini nella cartella 'images'
if __name__ == "__main__":
    images_dir = "images"  # Cartella nello stesso livello di predict.py
    if not os.path.exists(images_dir):
        print(f"❌ La cartella '{images_dir}' non esiste!")
        exit()
    
    # Trova tutti i file .jpg nella cartella
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    
    if not image_files:
        print(f"❌ Nessun file .jpg trovato nella cartella '{images_dir}'!")
        exit()
    
    print(f"📸 Trovate {len(image_files)} immagini da analizzare in '{images_dir}'")
    
    # Analizza ogni immagine
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        predict_image(img_path)
    
    print("\n✅ Analisi completata!")