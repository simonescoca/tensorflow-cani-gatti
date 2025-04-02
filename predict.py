import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import platform

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
        print(f"❌ Errore durante l'elaborazione dell'immagine: {str(e)}")
        exit()

# Funzione di predizione
def predict_image(img_path):
    img_array = prepare_image(img_path)
    prediction = model.predict(img_array)[0][0]  # Probabilità di "gatto"
    
    if prediction > 0.5:
        label = "Gatto"
        confidence = prediction * 100
        symbol = "🐱"
    else:
        label = "Cane"
        confidence = (1 - prediction) * 100
        symbol = "🐶"
    
    print(f"\n=== Risultato Predizione ===")
    print(f"{symbol} Predizione: {label}")
    print(f"Confidenza: {confidence:.2f}%")

# Esecuzione
if __name__ == "__main__":
    img_path = input("Inserisci il percorso dell'immagine: ").strip('"\' ')
    predict_image(img_path)