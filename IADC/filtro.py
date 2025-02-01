import asyncio
from bleak import BleakScanner, BleakClient
import struct
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import pywt
from scipy.signal import firwin, lfilter

# ========= CONFIGURACIÓN =========
TARGET_NAME = "Tensosense"
TARGET_UUID = "c4c1f6e2-4be5-11e5-885d-feff819cdc9f"
BUFFER_SIZE = 1024*2       # Muestras en el histórico
SAMPLING_RATE = 1024     # Hz (ajustar según dispositivo)
PLOT_REFRESH_RATE = 0.002  # Actualización gráfica cada 20ms

# ========= FILTROS =========
class KalmanFilter:
    """Filtro de Kalman para seguimiento en tiempo real"""
    def __init__(self):
        self.Q = 1e-5    # Varianza del proceso (ruido del sistema)
        self.R = 0.1**2  # Varianza de la medición (ruido del sensor)
        self.x = 15450   # Estado inicial (ajustar según señal base)
        self.P = 1.0     # Error inicial de estimación

    def update(self, z):
        # Predicción
        self.P += self.Q
        
        # Actualización
        K = self.P / (self.P + self.R)  # Ganancia de Kalman
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return self.x

# Configuración Wavelet
WAVELET_TYPE = 'sym6'    # Tipo de wavelet (sym6 para señales suaves)
WAVELET_LEVEL = 4        # Nivel de descomposición
WAVELET_WINDOW = 50      # Muestras para procesamiento por lotes

# Configuración FIR
FIR_TAPS = firwin(
    numtaps=31, 
    cutoff=2.0, 
    fs=SAMPLING_RATE, 
    window='blackmanharris'
)

# ========= INICIALIZACIÓN =========
# Buffers de datos
raw_buffer = deque(maxlen=BUFFER_SIZE)
kalman_buffer = deque(maxlen=BUFFER_SIZE)
wavelet_buffer = deque(maxlen=BUFFER_SIZE)
fir_buffer = deque(maxlen=BUFFER_SIZE)

# Objetos de filtrado
kf = KalmanFilter()
fir_samples = deque(maxlen=len(FIR_TAPS))

# Gráfico
plt.ion()
fig, ax = plt.subplots(figsize=(12,6))
ax.set_title("Comparación de Técnicas Avanzadas de Filtrado")
ax.set_xlabel("Muestras")
ax.set_ylabel("Valor ADC")
(line_raw,) = ax.plot([], [], label="Señal Cruda", alpha=0.5)
(line_kalman,) = ax.plot([], [], label="Kalman", linewidth=2)
(line_wavelet,) = ax.plot([], [], label="Wavelet", linestyle="--")
(line_fir,) = ax.plot([], [], label="FIR", color="black")
ax.legend()
ax.grid(True)

# ========= FUNCIONES DE PROCESAMIENTO =========
def apply_kalman(value):
    """Aplica filtro de Kalman en tiempo real"""
    filtered = kf.update(value)
    kalman_buffer.append(filtered)
    return filtered

def apply_wavelet_denoise():
    """Procesamiento por lotes con Wavelet"""
    if len(raw_buffer) >= WAVELET_WINDOW:
        # Descomposición Wavelet
        coeffs = pywt.wavedec(list(raw_buffer)[-WAVELET_WINDOW:], WAVELET_TYPE, level=WAVELET_LEVEL)
        
        # Umbralización adaptativa
        sigma = np.median(np.abs(coeffs[-WAVELET_LEVEL])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(WAVELET_WINDOW))
        
        # Aplicar umbral suave
        coeffs = [pywt.threshold(c, uthresh, 'soft') for c in coeffs]
        
        # Reconstrucción
        denoised = pywt.waverec(coeffs, WAVELET_TYPE)[-WAVELET_WINDOW:]
        wavelet_buffer.extend(denoised)

def apply_fir_filter(value):
    """Filtrado FIR con ventana deslizante"""
    fir_samples.append(value)
    if len(fir_samples) == len(FIR_TAPS):
        filtered = np.dot(FIR_TAPS, list(fir_samples))
        fir_buffer.append(filtered)

def update_plot():
    """Actualiza la visualización con todos los filtros"""
    line_raw.set_ydata(raw_buffer)
    line_raw.set_xdata(range(len(raw_buffer)))
    
    line_kalman.set_ydata(kalman_buffer)
    line_kalman.set_xdata(range(len(kalman_buffer)))
    
    line_wavelet.set_ydata(wavelet_buffer)
    line_wavelet.set_xdata(range(len(wavelet_buffer)))
    
    line_fir.set_ydata(fir_buffer)
    line_fir.set_xdata(range(len(fir_buffer)))
    
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(PLOT_REFRESH_RATE)

# ========= MANEJO BLE =========
def notification_handler(sender, data):
    try:
        # Decodificar valor ADC
        adc_value = struct.unpack('<H', data)[0]
        raw_buffer.append(adc_value)
        
        # Aplicar todos los filtros
        apply_kalman(adc_value)
        apply_wavelet_denoise()
        apply_fir_filter(adc_value)
        
        update_plot()
        
    except Exception as e:
        print(f"Error procesando datos: {str(e)}")

async def main():
    try:
        # Escanear dispositivo
        print("Buscando dispositivo Tensosense...")
        devices = await BleakScanner.discover(timeout=15)
        target = next((d for d in devices if d.name == TARGET_NAME), None)
        
        if not target:
            print("Dispositivo no encontrado")
            return

        # Conectar y configurar
        async with BleakClient(target.address) as client:
            print(f"Conectado a {target.name}")
            
            # Habilitar notificaciones
            await client.start_notify(TARGET_UUID, notification_handler)
            print("Capturando datos... (Presiona Ctrl+C para detener)")
            
            # Mantener conexión activa
            while True:
                await asyncio.sleep(1)
                
    except Exception as e:
        print(f"Error general: {str(e)}")
    finally:
        plt.close()

# ========= EJECUCIÓN =========
if __name__ == "__main__":
    # Instalar dependencias necesarias:
    # pip install bleak numpy pywavelets scipy matplotlib
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma terminado por el usuario")