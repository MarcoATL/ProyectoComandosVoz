import numpy as np
import librosa
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib
import pickle

# Definir la función de extracción de características
def extract_features(file_path):
    # Cargar el archivo de audio
    y, sr = librosa.load(file_path, duration=2.6)

    # Extraer MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_scaled = np.mean(mfcc.T,axis=0)
    
    return mfcc_scaled

def calcular_promedio_duracion(data_dir):
    duraciones = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                audio, sr = librosa.load(file_path)
                duracion = librosa.get_duration(y=audio, sr=sr)
                duraciones.append(duracion)
    
    promedio_duracion = np.mean(duraciones)
    
    return promedio_duracion

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
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(label_concat)


    # Convertir las listas a matrices numpy
    X = np.array(features)
    y = np.array(labels)
    print(y)
    return X, y


# Cargar los datos de entrenamiento y prueba
data_dir = "./AudiosComandosVoz"
print('Media de duracion de audios: '+ str(calcular_promedio_duracion(data_dir)) + ' segundos')
X, y = load_data(data_dir)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear el pipeline con escalado de características y clasificador SVM
pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

# Entrenar el modelo con los datos de entrenamiento
pipeline.fit(X_train, y_train)

# Evaluar el modelo en los datos de prueba
y_pred = pipeline.predict(X_test)

# Imprimir la matriz de confusión y el reporte de clasificación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado
joblib.dump(pipeline, 'modelo_entrenado.joblib')

'''
# Guardar el modelo entrenado
with open('modelo_entrenado.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
'''