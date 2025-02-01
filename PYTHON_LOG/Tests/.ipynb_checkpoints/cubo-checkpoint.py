import asyncio
from bleak import BleakScanner, BleakClient
import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- CONFIGURACIÓN ----------
TARGET_NAME = "Tensosense"  # Nombre del dispositivo
TARGET_UUID = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"  # UUID para recibir notificaciones

# Variables globales para orientación
orientation = [0, 0, 0]  # Roll, Pitch, Yaw


# ---------- FUNCIONES ----------
def update_orientation(x, y, z):
    """Actualizar orientación en grados (corregir límites)."""
    global orientation
    orientation[0] = (x / 100) % 360  # Roll (0-360 grados)
    orientation[1] = (y / 100) % 360  # Pitch (0-360 grados)
    orientation[2] = (z / 100) % 360  # Yaw (0-360 grados)


def notification_handler(sender, data):
    """Manejar datos BLE para actualizar orientación."""
    x, y, z = struct.unpack('<hhh', data)
    update_orientation(x, y, z)

    # Agregar impresión para depurar
    print(f"Orientación recibida -> Roll: {x / 100}, Pitch: {y / 100}, Yaw: {z / 100}")
    


async def connect_to_device():
    """Conectar al dispositivo BLE y habilitar notificaciones."""
    print("Buscando dispositivo BLE...")
    devices = await BleakScanner.discover(timeout=10)
    target_device = None

    # Mostrar todos los dispositivos detectados
    print("\nDispositivos encontrados:")
    for device in devices:
        print(f"Nombre: {device.name}, Dirección: {device.address}")

    # Buscar dispositivo por nombre
    for device in devices:
        if device.name == TARGET_NAME:
            target_device = device
            break

    if not target_device:
        print("Dispositivo no encontrado. Asegúrate de que esté encendido.")
        return None

    # Intentar conectar
    print(f"Intentando conectar a {TARGET_NAME} - {target_device.address}...")
    client = BleakClient(target_device.address)

    try:
        await client.connect()
        if client.is_connected:
            print("¡Conexión exitosa!")
            await client.start_notify(TARGET_UUID, notification_handler)
            return client
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None


# ---------- GRAFICAR ORIENTACIÓN ----------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Definir vértices de un cubo
vertices = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]
])

# Definir caras del cubo
faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], 
         [2, 3, 7, 6], [0, 4, 7, 3], [1, 5, 6, 2]]

def rotation_matrix(roll, pitch, yaw):
    """Crear matriz de rotación 3D basada en roll, pitch y yaw."""
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x

frame_count = 0  # Añadir contador global

def update(frame):
    """Actualizar visualización del cubo basado en orientación."""
    global frame_count
    frame_count += 1
    print(f"Frame actualizado: {frame_count}")  # Verificar actualización

    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualización de Orientación IMU')

    # Obtener ángulos actuales en radianes
    roll, pitch, yaw = np.radians(orientation)

    # Aplicar matriz de rotación
    R = rotation_matrix(roll, pitch, yaw)
    rotated_vertices = np.dot(vertices, R.T)

    # Dibujar caras
    poly3d = [[rotated_vertices[face] for face in f] for f in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', edgecolors='black', alpha=0.5))

ani = FuncAnimation(fig, update, interval=100, save_count=200)



# ---------- EJECUCIÓN ----------
async def main():
    """Conectar al dispositivo y mostrar el cubo."""
    client = await connect_to_device()
    if client:
        # Iniciar visualización
        ani = FuncAnimation(fig, update, interval=100)
        plt.show()
        # Mantener conexión activa
        while True:
            await asyncio.sleep(1)

# Ejecutar programa
asyncio.run(main())