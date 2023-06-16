import numpy as np
import librosa
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tempfile
import speech_recognition as sr
from pydub import AudioSegment

import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import joblib

def extract_features_a(audio_path):
    y, sr = librosa.load(audio_path)
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spect = librosa.power_to_db(spect, ref=np.max)

    # Redimensionar el espectrograma a una forma fija
    spect_resize = np.zeros((128, 938))
    spect_resize[:, :spect.shape[1]] = spect[:, :938]

    return spect_resize.T.flatten()  # Aplanar las características a un vector 1D

def record_audio(duration=5):
    r = sr.Recognizer()

    print("Iniciando grabación...")

    try:
        with sr.Microphone() as source:
            print("Grabando audio...")
            audio = r.record(source, duration=duration)
            print("Grabación finalizada")
            return audio
    except:
        print("Error al grabar audio")
        return None

def record_and_extract_features():
    audio = record_audio()

    if audio is None:
        return None

    # Convertir el objeto AudioData a un arreglo numpy
    audio_np = np.frombuffer(audio.frame_data, dtype=np.int16)

    # Asegurarse de que el audio tenga dos canales
    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=1)

    # Guardar el audio grabado en un archivo temporal
    temp_audio_path = "temp.wav"
    sf.write(temp_audio_path, audio_np, audio.sample_rate)

    # Extraer características del audio grabado
    print("Extrayendo características del audio...")
    features = extract_features_a(temp_audio_path)
    print("Forma de características:", features.shape)

    print("Extracción de características finalizada")

    return features.flatten()  # Aplanar las características a un vector 1D





# Definir la función de extracción de características
def extract_features(file_path):
    # Cargar el archivo de audio
    y, sr = librosa.load(file_path, duration=3.0)

    # Calcular el espectrograma
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convertir el espectrograma en decibeles
    spect_db = librosa.power_to_db(spect, ref=np.max)

    # Redimensionar el espectrograma a una forma fija
    spect_db_resize = np.zeros((128, 938))
    spect_db_resize[:, :spect_db.shape[1]] = spect_db[:, :938]

    # Aplanar el espectrograma en un vector 1D
    features = spect_db_resize.flatten()

    return features



# Definir la función de carga de datos
def load_data(data_dir):
    features = []
    labels = []

    # Recorrer los archivos de audio en el directorio de datos
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):  # Verificar si es un directorio
            for root, dirs, files in os.walk(label_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    # Concatenar los nombres de directorio en la etiqueta
                    label_concat = "-".join([label] + os.path.relpath(root, label_dir).split(os.sep))
                    # Extraer las características del archivo de audio
                    feature = extract_features(file_path=file_path)
                    features.append(feature)
                    labels.append(label_concat)


    # Convertir las listas a matrices numpy
    X = np.array(features)
    y = np.array(labels)
    print(y)
    return X, y






# Cargar los datos de entrenamiento y prueba
data_dir = "./AudiosComandosVoz"
X, y = load_data(data_dir)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir el clasificador K-NN
knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")

# Entrenar el clasificador con los datos de entrenamiento
knn.fit(X_train, y_train)
print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)

# Evaluar el modelo en los datos de prueba
y_pred = knn.predict(X_test)

# Guardar el modelo entrenado
joblib.dump(knn, 'modelo_entrenado.pkl')


joblib.dump(scaler, 'scaler.pkl')

# Cargar el modelo entrenado y el escalador desde los archivos
modelo = joblib.load('modelo_entrenado.pkl')
scaler = joblib.load('scaler.pkl')

def classify_audio_command(model_file, scaler_file):
    # Grabar el comando de voz
    features = record_and_extract_features()

    if features is None:
        return

    # Cargar el escalador previamente entrenado
    scaler = joblib.load(scaler_file)

    # Asegurarse de que las características sean una matriz 2D
    features = features.reshape(1, -1)

    # Escalar características de los datos de prueba
    features = scaler.transform(features)

    # Cargar el modelo previamente entrenado
    model = joblib.load(model_file)

    # Clasificar el audio grabado utilizando el modelo cargado
    prediction = model.predict(features)

    print("Comando clasificado:", prediction[0])



classify_audio_command('modelo_entrenado.pkl', 'scaler.pkl')
