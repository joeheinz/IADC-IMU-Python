import asyncio
from bleak import BleakScanner, BleakClient
import struct
import matplotlib.pyplot as plt
from collections import deque

# Target device details
TARGET_NAME = "Tensosense"
TARGET_UUID = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"  # UUID with 'notify' property

# Buffer size for plotting
BUFFER_SIZE = 100  # Número de puntos en la gráfica

# Datos para la gráfica (colas circulares)
x_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
y_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
z_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

# Crear la figura para la gráfica
plt.ion()  # Habilita el modo interactivo
fig, ax = plt.subplots()
line_x, = ax.plot(x_data, label="X")
line_y, = ax.plot(y_data, label="Y")
line_z, = ax.plot(z_data, label="Z")
ax.legend()
ax.set_ylim(-5000, 5000)  # Ajustar los límites del eje y
ax.set_title("IMU Data in Real Time")
ax.set_xlabel("Samples")
ax.set_ylabel("Acceleration (m/s²)")

# Función para actualizar la gráfica
def update_plot():
    line_x.set_ydata(x_data)
    line_y.set_ydata(y_data)
    line_z.set_ydata(z_data)
    plt.draw()
    plt.pause(0.01)

# Función para manejar notificaciones
def notification_handler(sender, data):
    # Decodifica los datos recibidos como 3 valores int16
    x, y, z = struct.unpack('<hhh', data)  # Little-endian, 3 int16
    print(f"Notification from {sender}: X={x}, Y={y}, Z={z}")

    # Añadir datos al buffer
    x_data.append(x)
    y_data.append(y)
    z_data.append(z)

    # Actualizar la gráfica
    update_plot()

# Función para escanear, conectar y escuchar notificaciones
async def scan_and_listen():
    try:
        print("Scanning for BLE devices...")

        # Escaneo de dispositivos
        devices = await BleakScanner.discover(timeout=10)
        target_device = None

        # Buscar dispositivo por nombre
        for device in devices:
            print(f"Name: {device.name}, Address: {device.address}, RSSI: {device.rssi} dBm")
            if device.name == TARGET_NAME:
                target_device = device
                print(f"\nFound target device: {device.name}, Address: {device.address}")
                break

        # Verificar si el dispositivo fue encontrado
        if not target_device:
            print(f"Device '{TARGET_NAME}' not found. Ensure it's powered on and advertising.")
            return

        # Conectar al dispositivo
        print(f"\nAttempting to connect to {target_device.address}...")
        async with BleakClient(target_device.address) as client:
            if client.is_connected:
                print(f"Connected to {target_device.name} at {target_device.address}!")

                # Descubrir servicios y características
                print("\nDiscovering services and characteristics...")
                for service in client.services:
                    print(f"[Service] {service.uuid}")
                    for char in service.characteristics:
                        print(f"  [Characteristic] {char.uuid}, Properties: {char.properties}")

                # Habilitar notificaciones
                print(f"\nEnabling notifications for UUID: {TARGET_UUID}")
                await client.start_notify(TARGET_UUID, notification_handler)

                print("\nListening for notifications. Press Ctrl+C to exit.")
                while True:
                    await asyncio.sleep(1)  # Mantener el programa ejecutándose

    except Exception as e:
        print(f"Error: {e}")

# Ejecutar la función asíncrona
asyncio.run(scan_and_listen())
