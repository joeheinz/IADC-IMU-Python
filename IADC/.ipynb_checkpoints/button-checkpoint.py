import asyncio
from bleak import BleakScanner, BleakClient

# 📌 Configura el nombre del dispositivo y UUID del GATT
TARGET_NAME = "Tensosense"  # Cambia esto según el nombre de tu dispositivo
TARGET_UUID = "61a885a4-41c3-60d0-9a53-6d652a70d29c"  # UUID para gattdb_automation_io

# Función para manejar notificaciones BLE
def notification_handler(sender, data):
    gpio_state = int.from_bytes(data, byteorder="little")
    print(f"🔵 GPIO PA0 State: {'ON' if gpio_state else 'OFF'}")

# Función para buscar y conectarse al dispositivo
async def scan_and_listen():
    print("🔍 Escaneando dispositivos BLE...")
    devices = await BleakScanner.discover(timeout=10)

    target_device = None
    for device in devices:
        print(f"🔗 Dispositivo encontrado: {device.name} [{device.address}]")
        if device.name == TARGET_NAME:
            target_device = device
            break

    if not target_device:
        print("❌ Dispositivo no encontrado.")
        return

    async with BleakClient(target_device.address) as client:
        print(f"✅ Conectado a {target_device.name}")

        # Activar notificaciones para el GATT de Automation IO
        await client.start_notify(TARGET_UUID, notification_handler)
        print("📡 Recibiendo notificaciones de GPIO PA0... (Ctrl+C para salir)")

        while True:
            await asyncio.sleep(1)  # Mantener la conexión activa

# Ejecutar el programa
asyncio.run(scan_and_listen())
