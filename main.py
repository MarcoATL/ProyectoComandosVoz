import numpy as np
import librosa
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Definir la función de extracción de características
def extract_features(file_path):
    # Cargar el archivo de audio
    y, sr = librosa.load(file_path, duration=2.55)

    # Calcular el espectrograma
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convertir el espectrograma en decibeles
    spect_db = librosa.power_to_db(spect, ref=np.max)


    # Normalizar el espectrograma
    spect_normalized = librosa.util.normalize(spect_db)

    # Redimensionar el espectrograma a una forma fija
    desired_shape = (128, 938)
    spect_db_resize = np.zeros(desired_shape)
    spect_normalized_resize = spect_normalized[:desired_shape[0], :desired_shape[1]]
    spect_db_resize[:spect_normalized_resize.shape[0], :spect_normalized_resize.shape[1]] = spect_normalized_resize
    
    ''' Lo del profe
    # Redimensionar el espectrograma a una forma fija
    spect_db_resize = np.zeros((128, 938))
    spect_db_resize[: spect_db.shape[0], : spect_db.shape[1]] = spect_db
    '''

    # Aplanar el espectrograma en un vector 1D
    features = spect_db_resize.flatten()

    return features



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
pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))

# Entrenar el modelo con los datos de entrenamiento
pipeline.fit(X_train, y_train)

# Evaluar el modelo en los datos de prueba
y_pred = pipeline.predict(X_test)

# Imprimir la matriz de confusión y el reporte de clasificación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

''' Lo del profe
# Escalado de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir el clasificador K-NN
knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")

# Entrenar el clasificador con los datos de entrenamiento
knn.fit(X_train, y_train)

# Evaluar el modelo en los datos de prueba
y_pred = knn.predict(X_test)

# Imprimir la matriz de confusión y el reporte de clasificación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''
