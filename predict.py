import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import platform

# Configurazione
print("\n=== Configurazione Prediction ===")
print(f"TensorFlow {tf.__version__}")
print(f"GPU disponibili: {tf.config.list_physical_devices('GPU')}\n")

# Carica modello
try:
    model = tf.keras.models.load_model('modello_animali_150x150.h5')
    print("✅ Modello caricato correttamente")
except Exception as e:
    print(f"❌ Errore caricamento modello: {str(e)}")
    exit()

# Funzione di preparazione immagine
def prepare_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        print(f"❌ Errore elaborazione immagine: {str(e)}")
        exit()

# Predizione
def predict_image(img_path):
    img_array = prepare_image(img_path)
    prediction = model.predict(img_array)
    confidence = prediction[0][0] * 100
    
    if prediction[0] > 0.5:
        print(f"🐶 Gatto (confidenza: {confidence:.2f}%)")
    else:
        print(f"🐱 Cane (confidenza: {100-confidence:.2f}%)")

# Esempio d'uso
if __name__ == "__main__":
    img_path = input("Inserisci il percorso dell'immagine: ").strip('"\' ')
    predict_image(img_path)