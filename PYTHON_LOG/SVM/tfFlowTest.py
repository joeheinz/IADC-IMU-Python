import tensorflow as tf
import numpy as np
import pandas as pd
import os

# ---------- CONFIGURACIÓN ----------
# Ruta del modelo y el dataset
MODEL_PATH = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/imu_gesture_model.h5'
DATASET_PATH = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration/tests'

# Parámetros del modelo
LABELS = ['Fall', 'Idle']  # Clases
SEQ_LENGTH = 70  # Número fijo de filas por muestra

# ---------- CARGAR MODELO ----------
# Cargar modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado con éxito:", model.input_shape)

# ---------- NORMALIZAR DATOS ----------
def normalize_data(data):
    """Normaliza los datos entre -1 y 1."""
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1


# ---------- PROCESAR CSV ----------
def preprocess_csv(file_path):
    """Preprocesa un archivo CSV para cumplir con el formato de entrada del modelo."""
    # Cargar CSV
    df = pd.read_csv(file_path)

    # Usar todas las columnas excepto Timestamp
    data = df.drop(columns=['Timestamp']).values

    # Asegurar exactamente 70 filas
    if len(data) < SEQ_LENGTH:
        # Rellenar filas faltantes repitiendo la última fila
        diff = SEQ_LENGTH - len(data)
        last_row = data[-1]
        padding = np.tile(last_row, (diff, 1))
        data = np.vstack((data, padding))
    else:
        # Tomar solo las primeras 70 filas
        data = data[:SEQ_LENGTH]

    # Normalizar entre -1 y 1
    data = normalize_data(data)

    # Dar formato para TensorFlow (agregar canal)
    data = data[..., np.newaxis]  # Shape: (70, 8, 1)
    return np.expand_dims(data, axis=0)  # Shape: (1, 70, 8, 1)


# ---------- FUNCIÓN DE PREDICCIÓN ----------
def predict_csv(file_path):
    """Predice la clase de un archivo CSV."""
    # Procesar el CSV
    input_data = preprocess_csv(file_path)
    print("Forma del input para predicción:", input_data.shape)

    # Realizar predicción
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions, axis=1)[0]

    # Mostrar resultados
    print("Predictions:", predictions)
    print("Predicted Label:", LABELS[predicted_label])


# ---------- EJECUCIÓN ----------
# Obtener un archivo CSV de la carpeta 'Fall'
csv_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.csv')]

if not csv_files:
    print("No se encontraron archivos CSV en la carpeta 'Fall'.")
else:
    # Seleccionar el primer archivo CSV para probar
    test_file = os.path.join(DATASET_PATH, csv_files[0])
    print("Probando archivo:", test_file)

    # Predecir clase
    predict_csv(test_file)