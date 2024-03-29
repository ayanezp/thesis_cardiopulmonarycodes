import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, TimeDistributed, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Adquisición de Datos: Carga de librerías y definición de estructuras básicas para el procesamiento de datos.

# Función para crear un espectrograma a partir de un archivo de audio
def create_spectrogram(wav_path):
    # Limpieza y Filtrado de Ruido: Creación de espectrogramas para visualizar la intensidad de frecuencia en función del tiempo.
    sample_rate, samples = wavfile.read(wav_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    return spectrogram

# Función para ajustar el tamaño de los espectrogramas
def adjust_spectrogram_size(spectrogram, max_length):
    # Normalización de Señales y Reducción del Tamaño de Espectrogramas: Ajuste del tamaño de los espectrogramas para estandarización.
    if spectrogram.shape[1] < max_length:
        padding = np.zeros((spectrogram.shape[0], max_length - spectrogram.shape[1]))
        spectrogram = np.hstack((spectrogram, padding))
    elif spectrogram.shape[1] > max_length:
        spectrogram = spectrogram[:, :max_length]
    return spectrogram

# Ruta a los archivos de audio y lectura de datos
audio_dir = 'C:/Users/Gus/Documents/archivos/Audio Files'
excel_path = 'C:/Users/Gus/Documents/archivos/Data annotation.xlsx'
df = pd.read_excel(excel_path)

# Extracción de información de nombres de archivos para etiquetado
def extraer_info_archivo(nombre_archivo):
    # Información Extraída de Nombres de Archivos: Extracción de metadatos relevantes de los nombres de archivos.
    partes = nombre_archivo.split('_')
    diagnostico, sound_type, location, age, gender = partes[1].split(',')[0], partes[1].split(',')[1], partes[1].split(',')[2], partes[1].split(',')[3], partes[1].split(',')[4].split('.')[0]
    return diagnostico, sound_type, location, age, gender

# Creación de un diccionario para mapear la información a los archivos de audio
mapeo_audio_etiqueta = {}
for archivo in os.listdir(audio_dir):
    if archivo.endswith('.wav'):
        diagnostico, sound_type, location, age, gender = extraer_info_archivo(archivo)
        etiqueta = df[(df['Diagnosis'] == diagnostico) & (df['Sound type'] == sound_type) & (df['Location'] == location) & (df['Age'] == int(age)) & (df['Gender'] == gender)]
        if not etiqueta.empty:
            mapeo_audio_etiqueta[archivo] = etiqueta.iloc[0]

# Filtrado y generación de espectrogramas para archivos con etiquetas correspondientes
espectrogramas_con_etiquetas = []
etiquetas_con_espectrogramas = []
for archivo, etiqueta in mapeo_audio_etiqueta.items():
    ruta_completa = os.path.join(audio_dir, archivo)
    espectrograma = create_spectrogram(ruta_completa)
    espectrogramas_con_etiquetas.append(espectrograma)
    etiquetas_con_espectrogramas.append(etiqueta['Diagnosis'])

# Encontrar el tamaño máximo de los espectrogramas y ajustarlos
max_length = max(spect.shape[1] for spect in espectrogramas_con_etiquetas)
adjusted_spectrograms = [adjust_spectrogram_size(spect, max_length) for spect in espectrogramas_con_etiquetas]

# Codificación de las etiquetas filtradas
le = LabelEncoder()
i_labels_filtradas = le.fit_transform(etiquetas_con_espectrogramas)
oh_labels_filtradas = to_categorical(i_labels_filtradas)

# Convertir los espectrogramas ajustados en un array de NumPy
X = np.array([np.expand_dims(spect, axis=-1) for spect in adjusted_spectrograms])
y = np.array(oh_labels_filtradas)

# División en conjuntos de entrenamiento y validación
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=6)

# Construcción del modelo con capas convolucionales y LSTM
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Reshape((-1, 64)),  # Ajuste para la capa LSTM
    LSTM(64, return_sequences=True),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(i_labels_filtradas)), activation='softmax')
])

# Configuración de optimizador y compilación del modelo
optimizer = Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Configuración de callbacks para el entrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min')

# Entrenamiento del modelo con callbacks
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stopping, reduce_lr])

# Visualización de espectrogramas y curvas de entrenamiento
# Pruebas Complementarias: Visualización de resultados y análisis de rendimiento del modelo.
def visualizar_espectrograma(spectrograma, titulo):
    plt.figure(figsize=(10, 4))
    plt.imshow(np.log(spectrograma + 1e-9), aspect='auto', cmap='viridis')
    plt.title(titulo)
    plt.xlabel('Tiempo')
    plt.ylabel('Frecuencia')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def visualizar_curvas_entrenamiento(history):
    plt.figure(figsize=[14, 10])
    plt.subplot(211)
    plt.plot(history.history['loss'], '#d62728', linewidth=3.0)
    plt.plot(history.history['val_loss'], '#1f77b4', linewidth=3.0)
    plt.legend(['Pérdida de Entrenamiento', 'Pérdida de Validación'], fontsize=18)
    plt.xlabel('Épocas', fontsize=16)
    plt.ylabel('Pérdida', fontsize=16)
    plt.title('Curvas de Pérdida', fontsize=16)

    plt.subplot(212)
    plt.plot(history.history['accuracy'], '#d62728', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], '#1f77b4', linewidth=3.0)
    plt.legend(['Precisión de Entrenamiento', 'Precisión de Validación'], fontsize=18)
    plt.xlabel('Épocas', fontsize=16)
    plt.ylabel('Precisión', fontsize=16)
    plt.title('Curvas de Precisión', fontsize=16)
    plt.show()

def visualizar_ejemplos_aleatorios(X, y, encoder, model):
    num_ejemplos = 3
    indices_ejemplos = random.sample(range(len(X)), num_ejemplos)
    for idx in indices_ejemplos:
        espectrograma = X[idx, :, :, 0]
        etiqueta_verdadera = encoder.classes_[np.argmax(y[idx])]
        prediccion = model.predict(np.expand_dims(X[idx], axis=0))[0]
        etiqueta_predicha = encoder.classes_[np.argmax(prediccion)]
        visualizar_espectrograma(espectrograma, f'Ejemplo - Etiqueta Verdadera: {etiqueta_verdadera}, Etiqueta Predicha: {etiqueta_predicha}')

visualizar_curvas_entrenamiento(history)
visualizar_ejemplos_aleatorios(X_valid, y_valid, le, model)

# Guardar el modelo entrenado
model.save("Segundo_Codigo")
