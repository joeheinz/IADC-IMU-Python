import asyncio
from bleak import BleakClient, BleakScanner

# Configuración
TARGET_NAME = "Tensosense"
CONNECTION_TIMEOUT = 30.0  # Aumentar timeout a 30 segundos

async def find_device():
    """Escanea con filtro por nombre y timeout extendido"""
    print("Buscando dispositivo...")
    try:
        device = await BleakScanner.find_device_by_name(
            TARGET_NAME,
            timeout=20.0  # Más tiempo para escanear
        )
        if device:
            print(f"Dispositivo encontrado: {device.name}, Dirección: {device.address}")
            return device.address
        print("Dispositivo no encontrado.")
        return None
    except Exception as e:
        print(f"Error escaneando: {e}")
        return None

async def list_services(address):
    """Conexión con timeout extendido y manejo de errores"""
    try:
        async with BleakClient(address, timeout=CONNECTION_TIMEOUT) as client:
            print("Conectado al dispositivo.")
            services = await client.get_services()
            print("\nServicios y características:")
            for service in services:
                print(f"\n[Servicio] {service.uuid}")
                for char in service.characteristics:
                    print(f"  └ Característica: {char.uuid}")
                    print(f"    ↳ Propiedades: {char.properties}")
    except Exception as e:
        print(f"\nError de conexión: {e}")

async def main():
    address = await find_device()
    if address:
        await list_services(address)

if __name__ == "__main__":
    asyncio.run(main())
