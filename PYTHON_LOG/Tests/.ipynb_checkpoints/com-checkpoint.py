import serial
import sys

def main():
    # Configuración del puerto serial
    port = '/dev/tty.usbmodem0004403122211'  # Reemplaza esto con el nombre de tu puerto
    baud_rate = 9600  # Ajusta según la configuración del dispositivo

    try:
        # Abre el puerto serial
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Escuchando en {port} a {baud_rate} baudios...")

        while True:
            # Lee una línea del puerto serial
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(line)

    except serial.SerialException as e:
        print(f"Error abriendo el puerto serial: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nTerminando programa...")
        ser.close()

if __name__ == "__main__":
    main()
