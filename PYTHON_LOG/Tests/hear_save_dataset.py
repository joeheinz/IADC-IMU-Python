import asyncio
from bleak import BleakScanner, BleakClient
import struct
import csv
import os
import time

# BLE Device Configuration
TARGET_NAME = "Tensosense"
TARGET_UUID = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"

# Output directory for datasets
output_dir = "/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/datasets"
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# Global variables
collecting = False
data_buffer = []

# Save dataset to CSV
def save_dataset():
    global data_buffer
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"dataset_{timestamp}.csv")

    # Write data to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "X", "Y", "Z"])  # Header
        writer.writerows(data_buffer)

    print(f"Dataset saved: {filename}")
    data_buffer = []  # Clear buffer

# Handle BLE notifications
def notification_handler(sender, data):
    global collecting, data_buffer

    # Decode 3 signed int16 values
    x, y, z = struct.unpack('<hhh', data)
    print(f"X={x}, Y={y}, Z={z}")

    # Collect data if recording
    if collecting:
        timestamp = time.time()
        data_buffer.append([timestamp, x, y, z])

# Scan, connect, and listen to notifications
async def scan_and_listen():
    global collecting

    try:
        # Scan for BLE devices
        print("Scanning for BLE devices...")
        devices = await BleakScanner.discover(timeout=10)
        target_device = None

        # Look for the target device
        for device in devices:
            print(f"Name: {device.name}, Address: {device.address}, RSSI: {device.rssi} dBm")
            if device.name == TARGET_NAME:
                target_device = device
                print(f"Found target device: {device.name}, Address: {device.address}")
                break

        if not target_device:
            print(f"Device '{TARGET_NAME}' not found. Ensure it's powered on.")
            return

        # Connect to the target device
        print(f"Connecting to {target_device.address}...")
        async with BleakClient(target_device.address) as client:
            if client.is_connected:
                print(f"Connected to {target_device.name} at {target_device.address}")

                # Enable notifications
                print(f"Enabling notifications for UUID: {TARGET_UUID}")
                await client.start_notify(TARGET_UUID, notification_handler)

                while True:
                    # Wait for the user to press Enter to start recording
                    input("\nPress ENTER to start recording...")
                    print("Starting in 3 seconds...")
                    for i in range(3, 0, -1):
                        print(f"{i}...")
                        await asyncio.sleep(1)

                    print("Recording for 5 seconds...")
                    collecting = True
                    await asyncio.sleep(5)  # Collect data for 5 seconds
                    collecting = False

                    # Save the dataset
                    save_dataset()
                    print("\nReady to record again. Press ENTER to continue...")

    except Exception as e:
        print(f"Error: {e}")

# Run the asynchronous function
asyncio.run(scan_and_listen())
