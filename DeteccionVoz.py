import sounddevice as sd
import numpy as np
import pickle
import librosa

# Cargar el modelo
with open('modelo_entrenado.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)

# Definir la función de extracción de características
def extract_features(y, sr):
    # Extraer MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    return mfcc_scaled.reshape(1, -1)  # reescalamos a 2D porque el modelo espera una entrada 2D

# Definir la función de grabación
def grabar_audio(duracion=2.5, fs=22050):
    grabacion = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()  # Esperar hasta que termine la grabación
    return grabacion.flatten()  # el audio se devuelve como un array 1D

# Función para predecir en tiempo real
def predecir_comando(modelo):
    print("Comienza grabación de audio...")
    grabacion = grabar_audio()
    print("Grabación terminada. Procesando...")

    # Extraer características y predecir
    features = extract_features(grabacion, 22050)
    pred = modelo.predict(features)[0]

    print("Predicción:", pred)


# Usar la función
predecir_comando(modelo)
