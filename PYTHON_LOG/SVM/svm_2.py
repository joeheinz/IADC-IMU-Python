import asyncio
from bleak import BleakScanner, BleakClient
import struct
import numpy as np
import pandas as pd
import os
import time

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.fft import fft, fftfreq


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

import joblib

# ---------- CONFIGURACIÓN ----------
# BLE Device Config
TARGET_NAME = "Tensosense"
UUID_ACC = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"   # UUID para aceleración
UUID_ORI = "b7c4b694-bee3-45dd-ba9f-f3b5e994f49a"   # UUID para orientación

# Ruta para guardar CSV
SAVE_PATH = "/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration"
os.makedirs(SAVE_PATH, exist_ok=True)

# Variables globales
raw_data = []  # Datos para guardar
acc_data = [0, 0, 0]  # Datos de aceleración
ori_data = [0, 0, 0]  # Datos de orientación

# ---------- FUNCIÓN PRINCIPAL ----------
def run_ble_capture():
    """Función principal para capturar datos IMU y guardarlos en CSV."""

    # ---------- NORMALIZACIÓN ----------
    def normalize_vector(x, y, z):
        """Calcula la magnitud normalizada de un vector."""
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        return magnitude / 1000  # Normalizado

    # ---------- MANEJAR DATOS DEL BLUETOOTH ----------
    def handle_acceleration(sender, data):
        """Manejar datos de aceleración."""
        global acc_data
        a_x, a_y, a_z = struct.unpack('<hhh', data)  # Decodificar aceleración
        acc_data = [a_x, a_y, a_z]  # Actualizar variables globales

    def handle_orientation(sender, data):
        """Manejar datos de orientación y guardar ambos datos."""
        global ori_data, acc_data, raw_data

        # Decodificar orientación
        o_x, o_y, o_z = struct.unpack('<hhh', data)
        ori_data = [o_x / 360, o_y / 360, o_z / 360]

        # Calcular 'Betrag' (magnitud) para ambas lecturas
        acc_betrag = normalize_vector(acc_data[0], acc_data[1], acc_data[2])
        ori_betrag = normalize_vector(o_x / 360, o_y / 360, o_z / 360)

        # Agregar datos al buffer
        timestamp = int(time.time() * 1000)  # Timestamp en milisegundos
        raw_data.append([
            acc_betrag, *acc_data,   # Aceleración
            ori_betrag, *ori_data,  # Orientación
            timestamp
        ])

        # Mostrar datos en consola
        print(f"Acc -> Betrag:{acc_betrag}, X:{acc_data[0]}, Y:{acc_data[1]}, Z:{acc_data[2]}")
        print(f"Ori -> Betrag:{ori_betrag}, X:{o_x}, Y:{o_y}, Z:{o_z}")

    # ---------- GUARDAR DATOS EN CSV ----------
    def save_to_csv():
        """Guarda los datos capturados en un archivo CSV."""
        global raw_data

        if not raw_data:
            print("Advertencia: No hay datos para guardar.")
            return

        # Crear DataFrame
        df = pd.DataFrame(raw_data, columns=[
            "Acc_Betrag", "Acc_X", "Acc_Y", "Acc_Z",  # Aceleración
            "Ori_Betrag", "Ori_X", "Ori_Y", "Ori_Z",  # Orientación
            "Timestamp"
        ])

        # Guardar CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(SAVE_PATH, f"data_{timestamp}.csv")
        df.to_csv(file_path, index=False)

        # Limpiar datos después de guardar
        raw_data = []
        print(f"Datos guardados en: {file_path}")
        return file_path  # Devolver el nombre del archivo guardado

    # ---------- FUNCIÓN PARA GRABAR DATOS ----------
    async def record_movement():
        """Graba datos durante 3 segundos."""
        global raw_data
        raw_data = []  # Limpiar datos anteriores

        try:
            print("\nBuscando dispositivo BLE...")
            # Buscar dispositivos BLE
            devices = await BleakScanner.discover(timeout=10)
            target_device = None
            for device in devices:
                if device.name == TARGET_NAME:
                    target_device = device
                    break

            if not target_device:
                print("Dispositivo no encontrado. Intenta nuevamente.")
                return

            # Conectar al dispositivo
            print("Conectando a Tensosense...")
            async with BleakClient(target_device.address) as client:
                if client.is_connected:
                    print("¡Conectado! Grabando por 3 segundos...")

                    # Activar notificaciones
                    await client.start_notify(UUID_ACC, handle_acceleration)
                    await client.start_notify(UUID_ORI, handle_orientation)

                    # Grabar durante 3 segundos
                    for i in range(3, 0, -1):
                        print(f"{i} segundos restantes...")
                        await asyncio.sleep(1)

                    # Detener notificaciones
                    await client.stop_notify(UUID_ACC)
                    await client.stop_notify(UUID_ORI)

                    # Guardar datos en CSV
                    file_path = save_to_csv()
                    print("¡Grabación completada!")
                    return file_path  # Devolver archivo guardado

        except Exception as e:
            print(f"Error: {e}")

    # ---------- INICIO ----------
    print("\nPresiona la barra espaciadora para comenzar la grabación.")
    while True:
        key = input()  # Esperar entrada de teclado
        if key == " ":  # Comenzar cuando se presiona la barra espaciadora
            return asyncio.run(record_movement())


        


            
            
            
"""FEATURE EXTRACTOR! ___________________"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.fft import fft, fftfreq

# ---------- CONFIGURACIÓN ----------
MODEL_PATH = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/imu_gesture_model.h5'
OUTPUT_FILE = 'feature_extraction_test.csv'
SEQ_LENGTH = 70  # Longitud fija
FEATURES = ['Acc_Betrag', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Ori_Betrag', 'Ori_X', 'Ori_Y', 'Ori_Z']

# Cargar modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado con éxito.")

# ---------- FUNCIÓN PARA EXTRAER CARACTERÍSTICAS ----------
def extract_features_from_file(csv_path):
    """Extrae características de un archivo CSV para usar en SVM."""
    # Leer CSV
    df = pd.read_csv(csv_path)
    
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
    predictions = model.predict(input_data)
    softmax_0 = predictions[0][0]  # Valor softmax para clase Fall
    softmax_1 = predictions[0][1]  # Valor softmax para clase Idle

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
        'File': os.path.basename(csv_path),
        'Softmax_Fall': softmax_0,
        'Softmax_Idle': softmax_1,
        **stats,
        'Acc_Betrag_Peak': acc_betrag_peak,
        'Ori_Betrag_Peak': ori_betrag_peak,
        'Acc_Peak_Frequency': peak_freq,
    }

    # Guardar resultados en CSV
    output_path = os.path.join(os.path.dirname(csv_path), OUTPUT_FILE)
    results_df = pd.DataFrame([features])
    if not os.path.exists(output_path):
        results_df.to_csv(output_path, index=False)
    else:
        results_df.to_csv(output_path, mode='a', header=False, index=False)

    print(f"Características guardadas en: {output_path}")
    return output_path
            
            

        
def svm(csv_file):
    # ---------- CARGAR DATOS ----------
    file_path = csv_file
    data = pd.read_csv(file_path)

    # Limpiar nombres de columnas
    data.columns = data.columns.str.strip()

    # Ver nombres de columnas
    print(data.columns)  # <-- Diagnóstico temporal

    # Preparar datos
    X = data[['Softmax_Fall', 'Softmax_Idle', 'Acc_Betrag_mu', 'Acc_Betrag_sigma',
          'Ori_Betrag_mu', 'Ori_Betrag_sigma', 'Acc_Peak_Frequency']].values

    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # Cargar modelo SVM
    loaded_svm = joblib.load('/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/svm_model.pkl')
    print("Modelo SVM cargado con éxito.")

    # Predicciones
    predictions_loaded = loaded_svm.predict(X_scaled[0:1])
    print("Predicción SVM:", predictions_loaded)


    """
if __name__ == "__main__":
    csv_file = run_ble_capture()
    if csv_file:
        output_path= extract_features_from_file(csv_file)
        svm(output_path)
        
   """
    
def main():
    while True:  # Bucle infinito
        # 1. Buscar BLE y capturar datos
        csv_file = run_ble_capture()

        if csv_file:
            # 2. Extraer características
            output_path = extract_features_from_file(csv_file)

            # 3. Ejecutar SVM
            svm(output_path)
            #svm('/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration/feature_extraction.csv')

            # 4. Borrar los archivos generados
            os.remove(csv_file)
            os.remove(output_path)

            print("Archivos temporales eliminados. Reiniciando proceso...\n")
            time.sleep(2)  # Esperar 2 segundos antes de reiniciar


if __name__ == "__main__":
    main()