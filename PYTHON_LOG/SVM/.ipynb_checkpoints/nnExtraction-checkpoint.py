import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- CONFIGURACIÓN ----------
# Ruta del dataset
DATASET_PATH = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration'
LABELS = ['FallEnlarged', 'IdleEnlarged']  # Etiquetas (clases)
SEQ_LENGTH = 70  # Número fijo de filas por muestra

# ---------- CARGAR DATOS ----------
def load_data():
    all_data = []
    all_labels = []

    # Leer cada carpeta (etiqueta)
    for label_idx, label_name in enumerate(LABELS):
        folder_path = os.path.join(DATASET_PATH, label_name)

        # Leer cada archivo CSV
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)

                # Leer CSV completo
                df = pd.read_csv(file_path)

                # Usar todas las columnas excepto Timestamp
                data = df.drop(columns=['Timestamp']).values  # Mantener todas las columnas excepto tiempo

                # **Asegurar exactamente 70 filas**
                if len(data) < SEQ_LENGTH:
                    # Rellenar filas faltantes repitiendo la última fila
                    diff = SEQ_LENGTH - len(data)
                    last_row = data[-1]
                    padding = np.tile(last_row, (diff, 1))
                    data = np.vstack((data, padding))
                else:
                    # Tomar solo las primeras 70 filas
                    data = data[:SEQ_LENGTH]

                # **Normalizar entre -1 y 1**
                data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1

                # Añadir datos y etiquetas
                all_data.append(data)
                all_labels.append(label_idx)

    # Convertir a arrays de NumPy
    all_data = np.array(all_data, dtype=np.float32)  # Shape: (n_samples, 70, 8)
    all_labels = np.array(all_labels, dtype=np.int32)  # Etiquetas

    return all_data, all_labels


# ---------- PREPROCESAR DATOS ----------
# Cargar los datos
data, labels = load_data()

# Dividir en conjuntos de entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Dar formato para TensorFlow (agregar canales)
x_train = x_train[..., np.newaxis]  # Shape: (n_samples, 70, 8, 1)
x_val = x_val[..., np.newaxis]

# Verificar formas
print("Forma de entrenamiento:", x_train.shape)
print("Forma de validación:", x_val.shape)

# Verificar contenido de datos
print("Ejemplo de datos (normalizados):", x_train[0][:5])  # Primeras 5 filas
print("Etiqueta correspondiente:", y_train[0])  # Mostrar etiqueta


# ---------- MODELO DE RED NEURONAL ----------
# ---------- MODELO DE RED NEURONAL ----------
# Crear modelo mejorado
# ---------- MODELO DE RED NEURONAL ----------
# Crear modelo mejorado
# ---------- MODELO DE RED NEURONAL ----------
model = tf.keras.Sequential([
    # Primera capa convolucional
    tf.keras.layers.Conv2D(8, (3, 2), activation='relu', padding='same',
                           input_shape=(70, 8, 1), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPool2D((2, 1)),
    tf.keras.layers.Dropout(0.5),  # Aumentado Dropout

    # Segunda capa convolucional
    tf.keras.layers.Conv2D(16, (3, 1), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPool2D((2, 1)),
    tf.keras.layers.Dropout(0.5),

    # Tercera capa convolucional (reducida)
    tf.keras.layers.Conv2D(32, (3, 1), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPool2D((2, 1)),
    tf.keras.layers.Dropout(0.5),

    # Aplanar salida
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),

    # Capa de salida
    tf.keras.layers.Dense(len(LABELS), activation='softmax')  # 2 clases
])

# Compilar modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reducimos el learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Mostrar resumen del modelo
model.summary()

# ---------- ENTRENAMIENTO ----------
from keras.callbacks import EarlyStopping

# Detener entrenamiento si la validación no mejora después de 5 épocas
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=80,  # Número máximo de épocas
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stop]  # Añadir EarlyStopping
)

# ---------- GUARDAR MODELO ----------
# Guardar en formato HDF5
model.save('imu_gesture_model.h5')

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar modelo TFLite
with open('imu_gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo guardado como TFLite.")