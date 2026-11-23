import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DEFINICIÓN DE LAS SEÑALES EN EL DOMINIO DEL TIEMPO
# ==============================================================================
fs = 1000.0                  # Frecuencia de muestreo (Hz)
T = 1.0 / fs                 # Periodo de muestreo (s)
t = np.arange(-1.0, 1.0, T)  # Vector de tiempo de -1 s a 1 s (N = 2000 muestras)

# a) Pulso rectangular (ancho = 0.2 s centrado en t = 0)
pulso_rect = np.where(np.abs(t) <= 0.1, 1.0, 0.0)

# b) Función escalón unitario u(t)
escalon = np.where(t >= 0.0, 1.0, 0.0)

# c) Función senoidal pura (f0 = 50 Hz, amplitud = 1)
f0 = 50.0
senoidal = np.sin(2 * np.pi * f0 * t)

# ==============================================================================
# 2. CÁLCULO DE LA TRANSFORMADA DE FOURIER (FFT)
# ==============================================================================
def calcular_espectro(senal, fs):
    """Calcula frecuencias centradas, magnitud normalizada y fase en radianes."""
    N = len(senal)
    fft_vals = np.fft.fftshift(np.fft.fft(senal))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    magnitud = np.abs(fft_vals) / N
    fft_limpia = np.where(np.abs(fft_vals) > 1e-6, fft_vals, 0)
    fase = np.angle(fft_limpia)
    return freqs, magnitud, fase

freqs, mag_rect, fase_rect = calcular_espectro(pulso_rect, fs)
_, mag_esc, fase_esc = calcular_espectro(escalon, fs)
_, mag_sen, fase_sen = calcular_espectro(senoidal, fs)

# ==============================================================================
# 3. VISUALIZACIÓN DE LOS RESULTADOS (TIEMPO, MAGNITUD Y FASE)
# ==============================================================================
senales = [
    ("Pulso Rectangular", pulso_rect, mag_rect, fase_rect, (-50, 50)),
    ("Función Escalón u(t)", escalon, mag_esc, fase_esc, (-50, 50)),
    ("Señal Senoidal (50 Hz)", senoidal, mag_sen, fase_sen, (-100, 100))
]

fig, axs = plt.subplots(3, 3, figsize=(13, 9.5))
fig.suptitle("Análisis en Dominio del Tiempo y Frecuencia (FFT)", fontsize=13, fontweight="bold")

for i, (nombre, x_t, mag, fase, xlim_f) in enumerate(senales):
    axs[i, 0].plot(t, x_t, color="#1E3A8A")
    axs[i, 0].set_title(f"{nombre} - Tiempo")
    axs[i, 0].set_xlabel("Tiempo (s)")
    axs[i, 0].set_ylabel("Amplitud")
    axs[i, 0].set_xlim(-0.5, 0.5)
    axs[i, 0].grid(True, alpha=0.3)

    axs[i, 1].plot(freqs, mag, color="#0F766E")
    axs[i, 1].set_title(f"{nombre} - Magnitud |X(f)|")
    axs[i, 1].set_xlabel("Frecuencia (Hz)")
    axs[i, 1].set_ylabel("Magnitud")
    axs[i, 1].set_xlim(xlim_f)
    axs[i, 1].grid(True, alpha=0.3)

    axs[i, 2].plot(freqs, fase, color="#B91C1C")
    axs[i, 2].set_title(f"{nombre} - Fase ∠X(f)")
    axs[i, 2].set_xlabel("Frecuencia (Hz)")
    axs[i, 2].set_ylabel("Fase (rad)")
    axs[i, 2].set_xlim(xlim_f)
    axs[i, 2].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("grafica_1_senales_elementales.png", dpi=300)
plt.show()

# ==============================================================================
# 4. ANÁLISIS DE PROPIEDADES DE LA TRANSFORMADA DE FOURIER
# ==============================================================================
# A) Linealidad
x1 = np.cos(2 * np.pi * 30 * t)
x2 = np.cos(2 * np.pi * 80 * t)
a, b = 2.0, 1.5
comb_tiempo = a * x1 + b * x2
_, mag_comb, _ = calcular_espectro(comb_tiempo, fs)
_, mag_x1, _ = calcular_espectro(x1, fs)
_, mag_x2, _ = calcular_espectro(x2, fs)
mag_suma_espectral = a * mag_x1 + b * mag_x2

# B) Desplazamiento en el tiempo
t0 = 0.15
pulso_desplazado = np.where(np.abs(t - t0) <= 0.1, 1.0, 0.0)
_, mag_desp, _ = calcular_espectro(pulso_desplazado, fs)

# C) Escalamiento en frecuencia/tiempo
pulso_comprimido = np.where(np.abs(2 * t) <= 0.1, 1.0, 0.0)
_, mag_esc_tiempo, _ = calcular_espectro(pulso_comprimido, fs)

fig2, axs2 = plt.subplots(1, 3, figsize=(13.5, 4.2))
fig2.suptitle("Verificación de Propiedades de la Transformada de Fourier", fontsize=12.5, fontweight="bold")

axs2[0].plot(freqs, mag_comb, label="FFT{a·x1 + b·x2}", lw=2)
axs2[0].plot(freqs, mag_suma_espectral, "--", label="a·FFT{x1} + b·FFT{x2}", lw=1.6)
axs2[0].set_title("1. Linealidad (30 Hz y 80 Hz)")
axs2[0].set_xlim(-120, 120)
axs2[0].set_xlabel("Frecuencia (Hz)")
axs2[0].legend()
axs2[0].grid(True, alpha=0.3)

axs2[1].plot(freqs, mag_rect, label="|X(f)| Original", lw=2)
axs2[1].plot(freqs, mag_desp, "--", label="|X(f)| Desplazado (t0=0.15s)", lw=1.6)
axs2[1].set_title("2. Desplazamiento en Tiempo")
axs2[1].set_xlim(-50, 50)
axs2[1].set_xlabel("Frecuencia (Hz)")
axs2[1].legend()
axs2[1].grid(True, alpha=0.3)

axs2[2].plot(freqs, mag_rect, label="Pulso original x(t)", lw=2)
axs2[2].plot(freqs, mag_esc_tiempo, label="Pulso comprimido x(2t)", lw=2)
axs2[2].set_title("3. Escalamiento (Tiempo vs. Frecuencia)")
axs2[2].set_xlim(-60, 60)
axs2[2].set_xlabel("Frecuencia (Hz)")
axs2[2].legend()
axs2[2].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("grafica_2_propiedades_fourier.png", dpi=300)
plt.show()
