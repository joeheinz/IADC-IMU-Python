import asyncio
from bleak import BleakScanner, BleakClient
import struct
import numpy as np
import pandas as pd
import os
from tkinter import *
from tkinter import messagebox
import time

# ---------- CONFIGURACIÓN ----------
# Nombre y UUID del dispositivo
TARGET_NAME = "Tensosense"
TARGET_UUID = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"

# Buffer para almacenar los datos
raw_data = []  # Datos para guardar en CSV

# Ruta para guardar CSV
SAVE_PATH = "/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration"
os.makedirs(SAVE_PATH, exist_ok=True)  # Crear carpeta si no existe


# ---------- NORMALIZAR ACELERACIÓN ----------
def normalize_acceleration(x, y, z):
    """Calcula la magnitud y la normaliza."""
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    return magnitude / 1000  # Normalizado a 1


# ---------- MANEJAR DATOS DEL BLUETOOTH ----------
def notification_handler(sender, data):
    """Procesa los datos BLE y actualiza el buffer."""
    global raw_data
    # Decodificar datos
    x, y, z = struct.unpack('<hhh', data)

    # Calcular la magnitud normalizada (Betrag)
    betrag = normalize_acceleration(x, y, z)

    # Obtener timestamp en milisegundos
    timestamp_ms = int(time.time() * 1000)

    # Guardar datos en el buffer
    raw_data.append([betrag, x, y, z, timestamp_ms])


# ---------- GUARDAR DATOS EN CSV ----------
def save_to_csv():
    """Guarda los datos capturados en un archivo CSV."""
    global raw_data

    if not raw_data:
        messagebox.showwarning("Advertencia", "No hay datos para guardar.")
        return

    # Crear DataFrame
    df = pd.DataFrame(raw_data, columns=["Betrag", "X", "Y", "Z", "Timestamp (ms)"])

    # Crear nombre de archivo
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(SAVE_PATH, f"data_{timestamp}.csv")

    # Guardar archivo
    df.to_csv(file_path, index=False)

    # Limpiar datos después de guardar
    raw_data = []
    messagebox.showinfo("Éxito", f"Datos guardados en:\n{file_path}")


# ---------- INTERFAZ GRÁFICA ----------
root = Tk()
root.title("Captura de Movimiento IMU")
root.geometry("400x300")

label_status = Label(root, text="Presiona 'Grabar Movimiento' para comenzar", font=("Arial", 12))
label_status.pack(pady=20)

label_timer = Label(root, text="", font=("Arial", 18))
label_timer.pack(pady=10)


# ---------- FUNCIÓN PARA GRABAR DATOS ----------
async def record_movement():
    """Graba datos durante 2 segundos."""
    global raw_data
    raw_data = []  # Limpiar datos anteriores

    try:
        label_status.config(text="Buscando dispositivo BLE...")
        root.update()

        # Buscar dispositivos BLE
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
                label_status.config(text="¡Conectado! Grabando por 2 segundos...")
                root.update()

                # Activar notificaciones
                await client.start_notify(TARGET_UUID, notification_handler)

                # Grabar durante 2 segundos
                for i in range(2, 0, -1):
                    label_timer.config(text=f"{i} segundos restantes...")
                    root.update()
                    await asyncio.sleep(1)

                # Detener notificaciones
                await client.stop_notify(TARGET_UUID)

                # Guardar los datos en CSV
                save_to_csv()
                label_status.config(text="¡Grabación completada!")
    except Exception as e:
        label_status.config(text=f"Error: {e}")


# ---------- FUNCIÓN PARA INICIAR GRABACIÓN ----------
def start_recording():
    asyncio.run(record_movement())

# Botón para grabar
btn_record = Button(root, text="Grabar Movimiento", command=start_recording, font=("Arial", 14))
btn_record.pack(pady=30)

# Ejecutar GUI
root.mainloop()