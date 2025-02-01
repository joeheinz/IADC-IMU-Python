import asyncio
from bleak import BleakScanner, BleakClient
import struct
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# Target device details
TARGET_NAME = "Tensosense"
TARGET_UUID = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"

# Buffer size for plotting
BUFFER_SIZE = 100

# Data for plotting (circular queues)
x_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
y_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
z_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
dx_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
dy_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
dz_data = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

# Create figure for the plot
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
line_x, = ax1.plot(x_data, label="X")
line_y, = ax1.plot(y_data, label="Y")
line_z, = ax1.plot(z_data, label="Z")
ax1.legend()
ax1.set_ylim(-5000, 5000)
ax1.set_title("IMU Acceleration Data in Real Time")
ax1.set_xlabel("Samples")
ax1.set_ylabel("Acceleration (m/s²)")

line_dx, = ax2.plot(dx_data, label="dX/dt")
line_dy, = ax2.plot(dy_data, label="dY/dt")
line_dz, = ax2.plot(dz_data, label="dZ/dt")
ax2.legend()
ax2.set_ylim(-5000, 5000)
ax2.set_title("IMU Acceleration Derivative")
ax2.set_xlabel("Samples")
ax2.set_ylabel("Velocity (m/s)")

# Function to update the plot
def update_plot():
    line_x.set_ydata(x_data)
    line_y.set_ydata(y_data)
    line_z.set_ydata(z_data)
    line_dx.set_ydata(dx_data)
    line_dy.set_ydata(dy_data)
    line_dz.set_ydata(dz_data)
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    plt.draw()
    plt.pause(0.01)

# Function to handle notifications
def notification_handler(sender, data):
    x, y, z = struct.unpack('<hhh', data)
    print(f"Notification from {sender}: X={x}, Y={y}, Z={z}")

    # Add acceleration data to buffer
    x_data.append(x)
    y_data.append(y)
    z_data.append(z)

    # Calculate derivative (simple approximation)
    if len(x_data) > 1:
        dx = x_data[-1] - x_data[-2]
        dy = y_data[-1] - y_data[-2]
        dz = z_data[-1] - z_data[-2]
        dx_data.append(dx)
        dy_data.append(dy)
        dz_data.append(dz)
    else:
        dx_data.append(0)
        dy_data.append(0)
        dz_data.append(0)

    update_plot()

# Function to scan, connect, and listen for notifications
async def scan_and_listen():
    try:
        print("Scanning for BLE devices...")

        # Scan for devices
        devices = await BleakScanner.discover(timeout=10)
        target_device = None

        for device in devices:
            print(f"Name: {device.name}, Address: {device.address}, RSSI: {device.rssi} dBm")
            if device.name == TARGET_NAME:
                target_device = device
                print(f"\nFound target device: {device.name}, Address: {device.address}")
                break

        if not target_device:
            print(f"Device '{TARGET_NAME}' not found. Ensure it's powered on and advertising.")
            return

        print(f"\nAttempting to connect to {target_device.address}...")
        async with BleakClient(target_device.address) as client:
            if client.is_connected:
                print(f"Connected to {target_device.name} at {target_device.address}!")

                print("\nDiscovering services and characteristics...")
                for service in client.services:
                    print(f"[Service] {service.uuid}")
                    for char in service.characteristics:
                        print(f"  [Characteristic] {char.uuid}, Properties: {char.properties}")

                print(f"\nEnabling notifications for UUID: {TARGET_UUID}")
                await client.start_notify(TARGET_UUID, notification_handler)

                print("\nListening for notifications. Press Ctrl+C to exit.")
                while True:
                    await asyncio.sleep(1)

    except Exception as e:
        print(f"Error: {e}")

# Run the async function
asyncio.run(scan_and_listen())