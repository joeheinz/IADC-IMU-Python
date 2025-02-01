
import asyncio
from bleak import BleakScanner, BleakClient
import struct
import numpy as np
import tensorflow as tf
from tkinter import *
from tkinter import messagebox
from collections import deque
import time

# ---------- CONFIGURACIÓN ----------
# Nombre del dispositivo BLE y UUID
TARGET_NAME = "Tensosense"
TARGET_UUID = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"

# Modelo y etiquetas
MODEL_PATH = "/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/Trained/imu_gesture_model.h5"  # Ruta al modelo
LABELS = ['Caidas', 'Sacudir', 'Quieto']  # Clases

# Parámetros del modelo
SEQ_LENGTH = 168
FEATURES = 3  # Número de características: X, Y, Z

# ---------- CARGAR MODELO ----------
# Cargar el modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)

# Crear un buffer para las secuencias
data_buffer = deque(maxlen=SEQ_LENGTH)

# ---------- INTERFAZ GRÁFICA (GUI) ----------
root = Tk()
root.title("Detector de Movimiento IMU")
root.geometry("400x300")

# Etiquetas y botones
label_status = Label(root, text="Presiona 'Grabar Movimiento' para comenzar", font=("Arial", 12))
label_status.pack(pady=20)

label_timer = Label(root, text="", font=("Arial", 18))
label_timer.pack(pady=10)

label_result = Label(root, text="", font=("Arial", 16))
label_result.pack(pady=20)

# ---------- FUNCIÓN PARA INFERENCIA ----------
def predict_movement():
    """Procesa los datos almacenados y predice el movimiento."""
    if len(data_buffer) < SEQ_LENGTH:
        return "Datos insuficientes..."

    # Convertir buffer a numpy array
    input_data = np.array(data_buffer).reshape(1, SEQ_LENGTH, FEATURES, 1)

    # Predecir
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions, axis=1)[0]  # Obtener el índice con mayor probabilidad
    return LABELS[predicted_label]

# ---------- FUNCIÓN DE NOTIFICACIÓN ----------
def notification_handler(sender, data):
    """Maneja los datos recibidos por Bluetooth."""
    x, y, z = struct.unpack('<hhh', data)  # Decodificar datos
    # Normalizar los datos
    x = x / 1000.0  # Asumiendo que los datos están entre -1000 y 1000
    y = y / 1000.0
    z = z / 1000.0
    # Añadir al buffer normalizado
    data_buffer.append([x, y, z])

# ---------- FUNCIÓN PARA GRABAR MOVIMIENTO ----------
async def record_movement():
    try:
        label_status.config(text="Buscando dispositivo Bluetooth...")
        root.update()

        # Escanear dispositivos
        devices = await BleakScanner.discover(timeout=10)
        target_device = None

        for device in devices:
            if device.name == TARGET_NAME:
                target_device = device
                break

        if not target_device:
            label_status.config(text="Dispositivo no encontrado. Intenta nuevamente.")
            return

        # Conectar al dispositivo
        label_status.config(text="Conectando...")
        root.update()

        async with BleakClient(target_device.address) as client:
            if client.is_connected:
                label_status.config(text="¡Conectado! Grabando por 6 segundos...")
                root.update()

                # Activar notificaciones
                await client.start_notify(TARGET_UUID, notification_handler)

                # Grabar datos por 5 segundos
                data_buffer.clear()
                for i in range(6, 0, -1):
                    label_timer.config(text=f"{i} segundos restantes...")
                    root.update()
                    await asyncio.sleep(1)

                # Detener grabación
                await client.stop_notify(TARGET_UUID)

                # Mostrar mensaje de procesamiento
                label_timer.config(text="Procesando datos...")
                root.update()

                # Predecir movimiento
                movement = predict_movement()
                label_result.config(text=f"Movimiento detectado: {movement}")
                label_status.config(text="¡Listo para grabar nuevamente!")
    except Exception as e:
        label_status.config(text=f"Error: {e}")

# ---------- FUNCIÓN PARA INICIAR GRABACIÓN ----------
def start_recording():
    asyncio.run(record_movement())

# Botón para iniciar grabación
btn_record = Button(root, text="Grabar Movimiento", command=start_recording, font=("Arial", 14))
btn_record.pack(pady=20)

# Ejecutar GUI
root.mainloop()
