import asyncio
from bleak import BleakClient, BleakScanner

# Configuración del dispositivo
TARGET_NAME = "Tensosense"
BATTERY_SERVICE_UUID = "1d14d6ee-fd63-4fa1-bfa4-8f47b42119f0"  # Servicio de batería
BATTERY_CHARACTERISTIC_UUID = "00002a58-0000-1000-8000-00805f9b34fb"  # Característica de nivel de batería

async def find_device():
    """Escanea dispositivos Bluetooth para encontrar el TARGET_NAME."""
    print("Buscando dispositivo...")
    devices = await BleakScanner.discover()
    for device in devices:
        if device.name == TARGET_NAME:
            print(f"Dispositivo encontrado: {device.name}, Dirección: {device.address}")
            return device.address
    print("Dispositivo no encontrado.")
    return None

async def read_battery_level_and_services(address):
    """Conecta al dispositivo, muestra los servicios y lee el nivel de batería."""
    async with BleakClient(address) as client:
        if not client.is_connected:
            print("No se pudo conectar al dispositivo.")
            return

        print("Conectado al dispositivo.")

        # Mostrar los servicios disponibles
        print("Servicios disponibles:")
        try:
            services = await client.get_services()
            for service in services:
                print(f"Servicio: {service.uuid}")
                for char in service.characteristics:
                    print(f"  Característica: {char.uuid} - Propiedades: {char.properties}")
        except Exception as e:
            print(f"No se pudieron obtener los servicios: {e}")

        # Leer el valor de la característica de nivel de batería
        try:
            battery_level = await client.read_gatt_char(BATTERY_CHARACTERISTIC_UUID)
            battery_percentage = int(battery_level[0])  # Valor en porcentaje
            print(f"Nivel Analogo: {battery_percentage}")
        except Exception as e:
            print(f"No se pudo leer el nivel de batería: {e}")

async def main():
    # Encuentra el dispositivo
    address = await find_device()
    if not address:
        return

    # Muestra los servicios y lee el nivel de batería
    await read_battery_level_and_services(address)

# Ejecuta la tarea principal
asyncio.run(main())