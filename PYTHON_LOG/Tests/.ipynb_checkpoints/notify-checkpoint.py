import asyncio
from bleak import BleakScanner, BleakClient

# Target device details
TARGET_NAME = "Blinky Example"
TARGET_UUID = "61a885a4-41c3-60d0-9a53-6d652a70d29c"  # UUID with 'notify' property

# Function to handle notifications
def notification_handler(sender, data):
    print(f"Notification from {sender}: {data}")

# Function to scan, connect, and listen to notifications
async def scan_and_listen():
    try:
        print("Scanning for BLE devices...")

        # Scan for devices
        devices = await BleakScanner.discover(timeout=10)
        target_device = None

        # Look for the device by name
        for device in devices:
            print(f"Name: {device.name}, Address: {device.address}, RSSI: {device.rssi} dBm")
            if device.name == TARGET_NAME:
                target_device = device
                print(f"\nFound target device: {device.name}, Address: {device.address}")
                break

        # Check if device was found
        if not target_device:
            print(f"Device '{TARGET_NAME}' not found. Ensure it's powered on and advertising.")
            return

        # Connect to the device
        print(f"\nAttempting to connect to {target_device.address}...")
        async with BleakClient(target_device.address) as client:
            if client.is_connected:
                print(f"Connected to {target_device.name} at {target_device.address}!")

                # Discover services and characteristics
                print("\nDiscovering services and characteristics...")
                for service in client.services:
                    print(f"[Service] {service.uuid}")
                    for char in service.characteristics:
                        print(f"  [Characteristic] {char.uuid}, Properties: {char.properties}")

                # Enable notifications on the specified UUID
                print(f"\nEnabling notifications for UUID: {TARGET_UUID}")
                await client.start_notify(TARGET_UUID, notification_handler)

                print("\nListening for notifications. Press Ctrl+C to exit.")
                while True:
                    await asyncio.sleep(1)  # Keep the program running and listening

    except Exception as e:
        print(f"Error: {e}")

# Run the async function
asyncio.run(scan_and_listen())
