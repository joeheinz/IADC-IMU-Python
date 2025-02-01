import asyncio
from bleak import BleakScanner, BleakClient
import struct
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

# ---------- CONFIGURACIÓN ----------
# BLE Device Config
TARGET_NAME = "Tensosense"
UUID_ACC = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"   # UUID para aceleración
UUID_ORI = "b7c4b694-bee3-45dd-ba9f-f3b5e994f49a"   # UUID para orientación

# Ruta para guardar CSV
SAVE_PATH = "/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/Fahrten"
os.makedirs(SAVE_PATH, exist_ok=True)

# Variables globales
raw_data = []  # Datos acumulados para guardar
acc_data = [0, 0, 0]  # Datos de aceleración
ori_data = [0, 0, 0]  # Datos de orientación
is_recording = False  # Estado de grabación


# ---------- FUNCIONES ----------
def normalize_vector(x, y, z):
    """Calcula la magnitud de un vector."""
    return np.sqrt(x**2 + y**2 + z**2)


def handle_acceleration(sender, data):
    """Manejar datos de aceleración."""
    global acc_data
    a_x, a_y, a_z = struct.unpack('<hhh', data)
    acc_data = [a_x, a_y, a_z]


def handle_orientation(sender, data):
    """Manejar datos de orientación y agregar a la lista."""
    global ori_data, acc_data, raw_data, is_recording

    # Decodificar orientación
    o_x, o_y, o_z = struct.unpack('<hhh', data)
    ori_data = [o_x, o_y, o_z]

    if is_recording:  # Guardar datos solo si está grabando
        timestamp = int(time.time() * 1000)  # Timestamp en milisegundos
        raw_data.append([
            timestamp,
            *acc_data,  # Aceleración: X, Y, Z
            normalize_vector(*acc_data),  # Betrag aceleración
            *ori_data,  # Orientación: X, Y, Z
            normalize_vector(*ori_data)  # Betrag orientación
        ])


async def save_to_csv():
    """Guarda los datos acumulados en un archivo CSV."""
    global raw_data

    if not raw_data:
        print("No hay datos para guardar.")
        return

    # Crear DataFrame
    df = pd.DataFrame(raw_data, columns=[
        "Timestamp",
        "Acc_X", "Acc_Y", "Acc_Z", "Acc_Betrag",
        "Ori_X", "Ori_Y", "Ori_Z", "Ori_Betrag"
    ])

    # Guardar en un archivo CSV con marca de tiempo
    filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    file_path = os.path.join(SAVE_PATH, filename)
    df.to_csv(file_path, index=False)

    # Vaciar datos acumulados
    raw_data = []
    print(f"Datos guardados en: {file_path}")


async def start_recording():
    """Inicia la grabación continua."""
    global is_recording
    is_recording = True

    try:
        print("\nBuscando dispositivo BLE...")
        devices = await BleakScanner.discover(timeout=10)
        target_device = None
        for device in devices:
            if device.name == TARGET_NAME:
                target_device = device
                break

        if not target_device:
            print("Dispositivo no encontrado. Intenta nuevamente.")
            return

        print("Conectando...")
        async with BleakClient(target_device.address) as client:
            if client.is_connected:
                print("¡Conectado! Grabando datos...")

                # Activar notificaciones para los sensores
                await client.start_notify(UUID_ACC, handle_acceleration)
                await client.start_notify(UUID_ORI, handle_orientation)

                while True:
                    # Guardar datos cada 5 minutos
                    await asyncio.sleep(300)  # 300 segundos = 5 minutos
                    await save_to_csv()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        is_recording = False
        await save_to_csv()  # Guardar cualquier dato restante antes de salir.


# ---------- FUNCIÓN PRINCIPAL ----------
def main():
    print("\nPresiona la barra espaciadora para comenzar la grabación. Presiona Ctrl+C para salir.")
    try:
        while True:
            key = input()
            if key == " ":  # Iniciar grabación al presionar espacio
                asyncio.run(start_recording())
    except KeyboardInterrupt:
        print("\nPrograma terminado por el usuario.")


# ---------- EJECUTAR ----------
if __name__ == "__main__":
    main()