import asyncio
from bleak import BleakScanner, BleakClient
import struct
import matplotlib.pyplot as plt
from collections import deque

# ---------- CONFIGURACIÓN ----------
# Nombre y UUID del dispositivo
TARGET_NAME = "Tensosense"
TARGET_UUID = "00002a58-0000-1000-8000-00805f9b34fb"  # UUID del IADC

# Parámetros de la gráfica
BUFFER_SIZE = 100  # Tamaño del buffer
adc_buffer = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)  # Cola circular para datos del ADC

# Crear gráfico
plt.ion()  # Modo interactivo
fig, ax = plt.subplots()
line_adc, = ax.plot(adc_buffer, label="Valor ADC")
ax.legend()
ax.set_ylim(0, 40)  # Ajustar el rango de la gráfica para valores de 12 bits
ax.set_title("Lectura del ADC en Tiempo Real")
ax.set_xlabel("Muestras")
ax.set_ylabel("Valor ADC")

# ---------- ACTUALIZAR GRÁFICO ----------
def update_plot():
    """Actualiza la gráfica en tiempo real."""
    line_adc.set_ydata(adc_buffer)
    line_adc.set_xdata(range(len(adc_buffer)))
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)

# ---------- FUNCIÓN PARA MANEJAR NOTIFICACIONES ----------
def notification_handler(sender, data):
    """
    Procesa los datos recibidos por BLE.
    """
    # Decodificar datos como un entero de 16 bits sin signo
    adc_value, = struct.unpack('<H', data)

    print(f"Valor ADC recibido: {adc_value}")

    # Actualizar buffer
    adc_buffer.append(adc_value)

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
