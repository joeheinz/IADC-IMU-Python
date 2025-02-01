import asyncio
from bleak import BleakScanner, BleakClient
import struct
import matplotlib.pyplot as plt
from collections import deque
import csv
import os
import time
import keyboard  # Para capturar teclas

# Configuración de dispositivo BLE
TARGET_NAME = "Tensosense"
TARGET_UUID = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"  # UUID con propiedad 'notify'

# Directorio donde se guardará el dataset
output_dir = "/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON\ LOG/datasets"
os.makedirs(output_dir, exist_ok=True)  # Crear carpeta si no existe

# Configuración de la gráfica
BUFFER_SIZE = 100
x_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
y_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
z_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

plt.ion()  # Modo interactivo
fig, ax = plt.subplots()
line_x, = ax.plot(x_data, label="X")
line_y, = ax.plot(y_data, label="Y")
line_z, = ax.plot(z_data, label="Z")
ax.legend()
ax.set_ylim(-5000, 5000)  # Limites del eje Y
ax.set_title("IMU Data in Real Time")
ax.set_xlabel("Samples")
ax.set_ylabel("Acceleration (m/s²)")

# Variables para el dataset
collecting = False
data_buffer = []

# Actualizar gráfica
def update_plot():
    line_x.set_ydata(x_data)
    line_y.set_ydata(y_data)
    line_z.set_ydata(z_data)
    plt.draw()
    plt.pause(0.01)

# Guardar el dataset en CSV
def save_dataset():
    global data_buffer
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Crear nombre único
    filename = os.path.join(output_dir, f"dataset_{timestamp}.csv")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "X", "Y", "Z"])  # Encabezado
        writer.writerows(data_buffer)
    print(f"Dataset guardado en: {filename}")
    data_buffer = []  # Limpiar buffer

# Manejar notificaciones BLE
def notification_handler(sender, data):
    global collecting, data_buffer

    # Decodificar datos como 3 valores int16
    x, y, z = struct.unpack('<hhh', data)
    print(f"Notification from {sender}: X={x}, Y={y}, Z={z}")

    # Añadir datos al buffer
    x_data.append(x)
    y_data.append(y)
    z_data.append(z)

    # Si estamos recolectando datos, añadirlos al buffer para guardar
    if collecting:
        timestamp = time.time()
        data_buffer.append([timestamp, x, y, z])

    # Actualizar gráfica
    update_plot()

# Proceso principal para conectar BLE y escuchar notificaciones
async def scan_and_listen():
    global collecting

    try:
        print("Escaneando dispositivos BLE...")
        devices = await BleakScanner.discover(timeout=10)
        target_device = None

        # Buscar dispositivo por nombre
        for device in devices:
            print(f"Name: {device.name}, Address: {device.address}, RSSI: {device.rssi} dBm")
            if device.name == TARGET_NAME:
                target_device = device
                print(f"\nDispositivo encontrado: {device.name}, Dirección: {device.address}")
                break

        # Verificar si se encontró el dispositivo
        if not target_device:
            print(f"Dispositivo '{TARGET_NAME}' no encontrado. Asegúrate de que esté encendido.")
            return

        # Conectar al dispositivo
        print(f"\nConectando a {target_device.address}...")
        async with BleakClient(target_device.address) as client:
            if client.is_connected:
                print(f"Conectado a {target_device.name} en {target_device.address}!")

                # Descubrir servicios y características
                print("\nDescubriendo servicios y características...")
                for service in client.services:
                    print(f"[Servicio] {service.uuid}")
                    for char in service.characteristics:
                        print(f"  [Característica] {char.uuid}, Propiedades: {char.properties}")

                # Habilitar notificaciones
                print(f"\nHabilitando notificaciones para UUID: {TARGET_UUID}")
                await client.start_notify(TARGET_UUID, notification_handler)

                print("\nEsperando notificaciones. Presiona ESPACIO para empezar a recolectar datos...")

                while True:
                    await asyncio.sleep(0.1)  # Mantener el programa en ejecución

                    # Detectar barra espaciadora para iniciar recolección
                    if keyboard.is_pressed('space'):
                        print("\n¡Preparado para grabar! Temporizador de 3 segundos:")
                        for i in range(3, 0, -1):
                            print(f"{i}...")
                            await asyncio.sleep(1)
                        print("¡Go!")

                        # Recolectar datos por 5 segundos
                        collecting = True
                        await asyncio.sleep(5)
                        collecting = False

                        # Guardar el dataset
                        save_dataset()
                        print("\nEsperando otra vez al teclado (ESPACIO)...")

    except Exception as e:
        print(f"Error: {e}")

# Ejecutar función principal
asyncio.run(scan_and_listen())
