import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Adquisición de Datos: Carga de librerías y preparación del conjunto de datos.
dataset = []
for folder in ["C:/Users/Gus/Documents/Python/Lightshot/set_a/*.wav", "C:/Users/Gus/Documents/Python/Lightshot/set_b/*.wav"]:
    for filename in glob.iglob(folder):
        if os.path.exists(filename):
            label = os.path.basename(filename).split("_")[0]
            duration = librosa.get_duration(filename=filename)
            # Filtrado de Datos por Duración: Descarte de archivos con duración menor a 3 segundos.
            if duration >= 3:
                slice_size = 3
                iterations = int((duration - slice_size) / (slice_size - 1))
                iterations += 1
                initial_offset = (duration - ((iterations * (slice_size - 1)) + 1)) / 2
                if label not in ["Aunlabelledtest", "Bunlabelledtest"]:
                    for i in range(iterations):
                        offset = initial_offset + i * (slice_size - 1)
                        dataset.append({"filename": filename, "label": label, "offset": offset})

dataset = pd.DataFrame(dataset)
dataset = shuffle(dataset, random_state=42)

# División en conjuntos de entrenamiento y prueba: Preparación de datos para el modelo.
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# Función para el aumento de datos (Data Augmentation)
def augment_data(y, sr):
    # Cambio de tono y velocidad para diversificar el conjunto de entrenamiento.
    y_changed = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-1, 2))
    y_changed = librosa.effects.time_stretch(y_changed, rate=np.random.uniform(0.9, 1.1))
    return y_changed

def extract_features(audio_path, offset, augment=False, max_pad_len=174):
    # Extracción de Características Espectrales: Uso de MFCCs, cromagrama y contraste espectral.
    y, sr = librosa.load(audio_path, offset=offset, duration=3)
    if augment:
        y = augment_data(y, sr)

    # Normalización de Señales: Asegurar longitud consistente de las señales.
    if len(y) < sr * 3:
        y = np.pad(y, (0, max(0, sr * 3 - len(y))), "constant")

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    # Reducción de Dimensionalidad: Ajuste de características mediante relleno o truncamiento.
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        chroma = np.pad(chroma, pad_width=((0, 0), (0, pad_width)), mode='constant')
        contrast = np.pad(contrast, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
        chroma = chroma[:, :max_pad_len]
        contrast = contrast[:, :max_pad_len]

    return np.concatenate((mfccs, chroma, contrast), axis=0)

# Preparación de los conjuntos de entrenamiento y prueba con y sin aumento de datos.
x_train = np.array([extract_features(row.filename, row.offset) for row in tqdm(train.itertuples(), total=len(train))])
x_test = np.array([extract_features(row.filename, row.offset) for row in tqdm(test.itertuples(), total=len(test))])
x_train_augmented = np.array([extract_features(row.filename, row.offset, augment=True) for row in tqdm(train.itertuples(), total=len(train))])
x_train = np.concatenate((x_train, x_train_augmented))

# Codificación de Etiquetas: Conversión de etiquetas de clase en valores numéricos.
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(train.label)
y_test_encoded = encoder.transform(test.label)
y_train = np.concatenate((y_train_encoded, y_train_encoded))

# Ponderación de Clases: Equilibrio de clases en el conjunto de datos.
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Preparación de datos para Keras: Ajuste de dimensiones para el modelo.
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train = keras.utils.to_categorical(y_train, num_classes=len(encoder.classes_))
y_test = keras.utils.to_categorical(y_test_encoded, num_classes=len(encoder.classes_))

# Construcción del Modelo de Machine Learning: Arquitectura CNN.
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
model.add(Dropout(0.5))
model.add(Dense(len(encoder.classes_), activation='softmax'))
model.summary()

# Compilación del Modelo: Configuración de optimizador y pérdida.
adam = Adam(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# Configuración de Early Stopping: Prevención de sobreajuste.
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Entrenamiento del Modelo: Uso de callbacks y ponderación de clases.
history = model.fit(x_train, y_train, batch_size=32, epochs=500, validation_data=(x_test, y_test), shuffle=True, callbacks=[early_stopping], class_weight=class_weights_dict)

# Visualización de Curvas de Pérdida y Precisión: Monitoreo del Rendimiento del Modelo.
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

# Evaluación y Predicciones del Modelo: Análisis de resultados.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Pérdida en Test:', scores[0])
print('Precisión en Test:', scores[1])
predictions = model.predict(x_test, verbose=1)
y_true = [encoder.classes_[np.argmax(y)] for y in y_test]
y_pred = [encoder.classes_[np.argmax(y)] for y in predictions]
print(classification_report(y_true, y_pred))

# Visualización de Matriz de Confusión: Evaluación detallada del rendimiento del modelo.
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
classes = encoder.classes_
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Verdaderas')
plt.show()

# Curvas ROC y Precisión-Recall (Opcional): Análisis de rendimiento en clasificación binaria.
if len(encoder.classes_) == 2:
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()
    # Curva de Precisión y Recall
    precision, recall, _ = precision_recall_curve(y_true, y_pred[:, 1])
    aps = average_precision_score(y_true, y_pred[:, 1])
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Curva de Precisión y Recall (AP={aps:.2f})')
    plt.show()

# Visualización de Espectrogramas de Ejemplo: Interpretación de resultados.
num_examples = 3
for i in range(num_examples):
    index = np.random.randint(0, len(x_test))
    sample = x_test[index]
    prediction = model.predict(np.expand_dims(sample, axis=0))[0]
    predicted_label = encoder.classes_[np.argmax(prediction)]
    true_label = encoder.classes_[np.argmax(y_test[index])]
    plt.figure(figsize=(12, 4))
    plt.title(f'Ejemplo {i+1} - Predicción: {predicted_label}, Etiqueta Verdadera: {true_label}')
    librosa.display.specshow(librosa.amplitude_to_db(sample[:, :, 0], ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

# Guardar el Modelo: Preservación del modelo entrenado.
model.save("Primer_codigo.h5")
