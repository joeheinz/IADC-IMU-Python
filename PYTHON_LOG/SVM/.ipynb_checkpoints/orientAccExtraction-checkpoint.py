import asyncio
from bleak import BleakScanner, BleakClient
import struct
import numpy as np
import pandas as pd
import os
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk  # Para el GIF animado
import time

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
    ori_data = [o_x/360, o_y/360, o_z/360]

    # Calcular 'Betrag' (magnitud) para ambas lecturas
    acc_betrag = normalize_vector(acc_data[0], acc_data[1], acc_data[2])
    ori_betrag = normalize_vector(o_x/360, o_y/360, o_z/360)

    # Agregar datos al buffer
    timestamp = int(time.time() * 1000)  # Timestamp en milisegundos
    raw_data.append([
        acc_betrag, *acc_data,   # Aceleración
        ori_betrag, *ori_data,  # Orientación
        timestamp
    ])

    # Mostrar datos en consola (para depuración)
    print(f"Acc -> Betrag:{acc_betrag}, X:{acc_data[0]}, Y:{acc_data[1]}, Z:{acc_data[2]}")
    print(f"Ori -> Betrag:{ori_betrag}, X:{o_x}, Y:{o_y}, Z:{o_z}")


# ---------- GUARDAR DATOS EN CSV ----------
def save_to_csv():
    """Guarda los datos capturados en un archivo CSV."""
    global raw_data

    if not raw_data:
        messagebox.showwarning("Advertencia", "No hay datos para guardar.")
        return

    # Crear DataFrame con columnas separadas para aceleración y orientación
    df = pd.DataFrame(raw_data, columns=[
        "Acc_Betrag", "Acc_X", "Acc_Y", "Acc_Z",  # Aceleración
        "Ori_Betrag", "Ori_X", "Ori_Y", "Ori_Z",  # Orientación
        "Timestamp"
    ])

    # Guardar el archivo CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(SAVE_PATH, f"data_{timestamp}.csv")
    df.to_csv(file_path, index=False)

    # Limpiar datos después de guardar
    raw_data = []
    messagebox.showinfo("Éxito", f"Datos guardados en:\n{file_path}")


# ---------- INTERFAZ GRÁFICA ----------
root = Tk()
root.title("Captura de Movimiento IMU")
root.geometry("700x500")

# ---------- CARGAR GIF ----------
# Ruta al GIF
GIF_PATH = os.path.join(os.path.dirname(__file__), "gif.gif")  # Cambia el nombre si tu archivo es diferente

if os.path.exists(GIF_PATH):
    # Cargar frames del GIF
    gif_image = Image.open(GIF_PATH)
    frames = []

    try:
        while True:
            frame = gif_image.copy()
            frames.append(ImageTk.PhotoImage(frame))
            gif_image.seek(len(frames))  # Siguiente frame
    except EOFError:
        pass  # Fin del GIF

    # Mostrar el primer frame
    gif_label = Label(root)
    gif_label.pack(pady=10)

    def update_gif(index=0):
        """Actualizar frames del GIF."""
        frame = frames[index]
        gif_label.config(image=frame)
        index = (index + 1) % len(frames)
        root.after(100, update_gif, index)

    update_gif()
else:
    messagebox.showerror("Error", "GIF no encontrado. Asegúrate de tener 'animation.gif' en el directorio.")

# ---------- ETIQUETAS ----------
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

                # Activar notificaciones para aceleración y orientación
                await client.start_notify(UUID_ACC, handle_acceleration)
                await client.start_notify(UUID_ORI, handle_orientation)

                # Grabar durante 2 segundos
                for i in range(2, 0, -1):
                    label_timer.config(text=f"{i} segundos restantes...")
                    root.update()
                    await asyncio.sleep(1)

                # Detener notificaciones
                await client.stop_notify(UUID_ACC)
                await client.stop_notify(UUID_ORI)

                # Guardar los datos en CSV
                save_to_csv()
                label_status.config(text="¡Grabación completada!")
    except Exception as e:
        label_status.config(text=f"Error: {e}")


# ---------- BOTÓN ----------
def start_recording():
    asyncio.run(record_movement())

btn_record = Button(root, text="Grabar Movimiento", command=start_recording, font=("Arial", 14))
btn_record.pack(pady=20)

root.mainloop()