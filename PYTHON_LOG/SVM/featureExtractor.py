import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.fft import fft, fftfreq

# ---------- CONFIGURACIÓN ----------
# Directorios
DATASET_PATH = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration'
LABELS = ['FallEnlarged', 'IdleEnlarged']  # Directorios
MODEL_PATH = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/imu_gesture_model.h5'
OUTPUT_FILE = 'feature_extraction.csv'

# Parámetros
SEQ_LENGTH = 70  # Longitud fija
FEATURES = ['Acc_Betrag', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Ori_Betrag', 'Ori_X', 'Ori_Y', 'Ori_Z']

# Cargar modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado con éxito.")

# ---------- FUNCIÓN PARA EXTRAER CARACTERÍSTICAS ----------
def extract_features(file_path, label):
    # Leer CSV
    df = pd.read_csv(file_path)
    
    # Normalizar datos (entre -1 y 1)
    data = df.drop(columns=['Timestamp']).values
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1

    # Preparar input para el modelo
    if len(data) < SEQ_LENGTH:
        # Rellenar filas faltantes
        diff = SEQ_LENGTH - len(data)
        last_row = data[-1]
        padding = np.tile(last_row, (diff, 1))
        data = np.vstack((data, padding))
    else:
        # Tomar solo las primeras 70 filas
        data = data[:SEQ_LENGTH]
    
    # Dar formato al input (1, 70, 8, 1)
    input_data = data[..., np.newaxis]
    input_data = np.expand_dims(input_data, axis=0)
    
    # Predicción
    predictions = model.predict(input_data)  # Vector softmax con 2 valores
    predicted_label = np.argmax(predictions)
    predicted_class = LABELS[predicted_label]
    softmax_0 = predictions[0][0]  # Valor softmax para clase 0 (FallEnlarged)
    softmax_1 = predictions[0][1]  # Valor softmax para clase 1 (IdleEnlarged)

    # Calcular estadísticas (μ y σ)
    stats = {}
    for i, feature in enumerate(FEATURES):
        stats[f"{feature}_mu"] = np.mean(data[:, i])
        stats[f"{feature}_sigma"] = np.std(data[:, i])

    # Picos
    acc_betrag_peak = np.max(data[:, 0])  # Pico de Aceleración
    ori_betrag_peak = np.max(data[:, 4])  # Pico de Orientación

    # FFT - Frecuencia pico de Aceleración
    acc_signal = data[:, 0]  # Aceleración
    fft_values = np.abs(fft(acc_signal))
    freqs = fftfreq(SEQ_LENGTH, d=0.01)  # Suponiendo 100 Hz (10 ms)
    peak_freq = freqs[np.argmax(fft_values[1:])]  # Ignorar componente DC

    # Crear el feature set
    features = {
        'File': os.path.basename(file_path),
        'Predicted_Label': predicted_class,
        'Softmax_Fall': softmax_0,  # Output softmax clase Fall
        'Softmax_Idle': softmax_1,  # Output softmax clase Idle
        **stats,
        'Acc_Betrag_Peak': acc_betrag_peak,
        'Ori_Betrag_Peak': ori_betrag_peak,
        'Acc_Peak_Frequency': peak_freq,
        'Label': label  # Etiqueta del directorio
    }
    return features


# ---------- PROCESAR LOS DATOS ----------
results = []

# Procesar todos los CSV en los directorios
for label in LABELS:
    folder_path = os.path.join(DATASET_PATH, label)
    count = 0  # Contador para archivos procesados

    for file in os.listdir(folder_path):
        if file.endswith('.csv') and count < 500:  # Procesar solo 500 archivos
            file_path = os.path.join(folder_path, file)
            print(f"Procesando: {file_path}")

            # Extraer características
            features = extract_features(file_path, label)
            results.append(features)
            count += 1

        if count >= 500:  # Limitar a 500 archivos por directorio
            break

# Guardar resultados
results_df = pd.DataFrame(results)
output_path = os.path.join(DATASET_PATH, OUTPUT_FILE)
results_df.to_csv(output_path, index=False)
print(f"Características guardadas en: {output_path}")