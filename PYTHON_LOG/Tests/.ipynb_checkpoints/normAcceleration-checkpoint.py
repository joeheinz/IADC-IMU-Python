import asyncio
from bleak import BleakScanner, BleakClient
import struct
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# ---------- CONFIGURACIÓN ----------
# Nombre y UUID del dispositivo
TARGET_NAME = "Tensosense"
TARGET_UUID = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"

# Parámetros de la gráfica
BUFFER_SIZE = 100  # Tamaño del buffer
acc_buffer = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)  # Cola circular para datos normalizados

# Crear gráfico
plt.ion()  # Modo interactivo
fig, ax = plt.subplots()
line_acc, = ax.plot(acc_buffer, label="Aceleración Normalizada")
ax.legend()
ax.set_ylim(0, 3.5)  # Rango de la gráfica
ax.set_title("Aceleración Normalizada en Tiempo Real")
ax.set_xlabel("Muestras")
ax.set_ylabel("Valor Normalizado")

# ---------- FUNCIÓN PARA NORMALIZACIÓN ----------
def normalize_acceleration(x, y, z):
    """
    Calcula la magnitud absoluta de la aceleración y la normaliza.
    Fórmula: ||a|| = sqrt(x^2 + y^2 + z^2)
    """
    # 1. Calcular la magnitud
    magnitude = np.sqrt(x**2 + y**2 + z**2)


    # 3. Retornar la magnitud normalizada
    return np.sqrt(x**2 + y**2 + z**2) / 1000


# ---------- ACTUALIZAR GRÁFICO ----------
def update_plot():
    """Actualiza la gráfica en tiempo real."""
    line_acc.set_ydata(acc_buffer)
    plt.draw()
    plt.pause(0.01)


# ---------- FUNCIÓN PARA MANEJAR NOTIFICACIONES ----------
def notification_handler(sender, data):
    """
    Procesa los datos recibidos por BLE.
    """
    # Decodificar datos como 3 enteros int16 (X, Y, Z)
    x, y, z = struct.unpack('<hhh', data)

    # Calcular y normalizar la aceleración
    normalized_acc = normalize_acceleration(x, y, z)
    print(f"X={x}, Y={y}, Z={z} -> Magnitud Normalizada: {normalized_acc:.4f}")

    # Actualizar buffer
    acc_buffer.append(normalized_acc)

    # Actualizar gráfico
    update_plot()


# ---------- FUNCIÓN PARA ESCANEAR Y ESCUCHAR ----------
async def scan_and_listen():
    try:
        print("Escaneando dispositivos BLE...")

        # Escanear dispositivos
        devices = await BleakScanner.discover(timeout=10)
        target_device = None

        # Buscar el dispositivo por nombre
        for device in devices:
            print(f"Nombre: {device.name}, Dirección: {device.address}")
            if device.name == TARGET_NAME:
                target_device = device
                print(f"\n¡Dispositivo encontrado! Conectando a {device.address}")
                break

        # Si no se encuentra el dispositivo
        if not target_device:
            print(f"Dispositivo '{TARGET_NAME}' no encontrado.")
            return

        # Conectar al dispositivo
        async with BleakClient(target_device.address) as client:
            if client.is_connected:
                print(f"Conectado a {target_device.name} ({target_device.address})")

                # Descubrir servicios y características
                for service in client.services:
                    print(f"[Servicio] {service.uuid}")
                    for char in service.characteristics:
                        print(f"  [Característica] {char.uuid}, Propiedades: {char.properties}")

                # Iniciar notificaciones
                print(f"\nActivando notificaciones para UUID: {TARGET_UUID}")
                await client.start_notify(TARGET_UUID, notification_handler)

                print("Recibiendo datos. Presiona Ctrl+C para salir...")
                while True:
                    await asyncio.sleep(1)  # Mantener el programa ejecutándose

    except Exception as e:
        print(f"Error: {e}")


# ---------- EJECUTAR PROGRAMA ----------
asyncio.run(scan_and_listen())