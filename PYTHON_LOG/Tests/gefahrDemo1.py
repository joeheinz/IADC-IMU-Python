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

# Advertencias
warning_text = None
oscillation_text = None
high_acceleration_text = None

# Variables para detección de oscilaciones
oscillation_threshold = 1.0  # Umbral para detectar oscilaciones
oscillation_detected = False
prev_acc = 0  # Almacena el valor anterior para comparar cambios rápidos


# ---------- FUNCIÓN PARA NORMALIZACIÓN ----------
def normalize_acceleration(x, y, z):
    """
    Calcula la magnitud absoluta de la aceleración y la normaliza.
    """
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    return magnitude / 1000


# ---------- DETECCIÓN DE EVENTOS ----------
def check_free_fall(normalized_acc):
    """
    Detecta si el dispositivo está en caída libre.
    """
    return normalized_acc < 0.1  # Umbral para caída libre


def detect_oscillations(normalized_acc):
    """
    Detecta oscilaciones bruscas en la aceleración.
    """
    global oscillation_detected, prev_acc

    # Detectar cambios rápidos por encima o por debajo del umbral
    if (prev_acc > oscillation_threshold and normalized_acc < oscillation_threshold) or \
       (prev_acc < oscillation_threshold and normalized_acc > oscillation_threshold):
        oscillation_detected = True
    else:
        oscillation_detected = False

    # Actualizar el valor previo
    prev_acc = normalized_acc

    return oscillation_detected


def detect_high_acceleration(normalized_acc):
    """
    Detecta aceleraciones fuertes por encima de 2.
    """
    return normalized_acc > 2.0  # Umbral para aceleraciones fuertes


# ---------- ACTUALIZAR GRÁFICO ----------
def update_plot_with_oscillation(is_free_fall, is_oscillating, is_high_acceleration):
    """
    Actualiza la gráfica en tiempo real y muestra advertencias si es necesario.
    """
    global warning_text, oscillation_text, high_acceleration_text

    # Advertencia de caída libre (en rojo)
    if is_free_fall:
        if warning_text is None:
            warning_text = ax.text(10, 3.0, "Vorsicht! Gerät im freien Fall!", color='red', fontsize=12, fontweight='bold')
    else:
        if warning_text is not None:
            warning_text.remove()
            warning_text = None

    # Advertencia de oscilaciones (en azul)
    if is_oscillating:
        if oscillation_text is None:
            oscillation_text = ax.text(10, 2.5, "Vorsicht! Gerät wird geschüttelt!", color='blue', fontsize=12, fontweight='bold')
    else:
        if oscillation_text is not None:
            oscillation_text.remove()
            oscillation_text = None

    # Advertencia de aceleraciones fuertes (en rojo)
    if is_high_acceleration:
        if high_acceleration_text is None:
            high_acceleration_text = ax.text(10, 2.0, "Starke Beschleunigungen!", color='red', fontsize=12, fontweight='bold')
    else:
        if high_acceleration_text is not None:
            high_acceleration_text.remove()
            high_acceleration_text = None

    # Actualizar gráfica
    line_acc.set_ydata(acc_buffer)
    plt.draw()
    plt.pause(0.01)


# ---------- FUNCIÓN PARA MANEJAR NOTIFICACIONES ----------
def notification_handler_with_oscillation(sender, data):
    """
    Procesa los datos recibidos por BLE e incluye detección de eventos.
    """
    # Decodificar datos como 3 enteros int16 (X, Y, Z)
    x, y, z = struct.unpack('<hhh', data)

    # Calcular y normalizar la aceleración
    normalized_acc = normalize_acceleration(x, y, z)
    print(f"X={x}, Y={y}, Z={z} -> Magnitud Normalizada: {normalized_acc:.4f}")

    # Actualizar buffer
    acc_buffer.append(normalized_acc)

    # Detectar eventos
    is_free_fall = check_free_fall(normalized_acc)  # Caída libre
    is_oscillating = detect_oscillations(normalized_acc)  # Oscilaciones
    is_high_acceleration = detect_high_acceleration(normalized_acc)  # Aceleración fuerte

    # Actualizar gráfico con advertencias
    update_plot_with_oscillation(is_free_fall, is_oscillating, is_high_acceleration)


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

                # Iniciar notificaciones
                await client.start_notify(TARGET_UUID, notification_handler_with_oscillation)

                print("Recibiendo datos. Presiona Ctrl+C para salir...")
                while True:
                    await asyncio.sleep(1)

    except Exception as e:
        print(f"Error: {e}")


# ---------- EJECUTAR PROGRAMA ----------
asyncio.run(scan_and_listen())