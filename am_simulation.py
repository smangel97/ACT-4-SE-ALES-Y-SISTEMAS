"""Simula AM convencional y analiza ruido, atenuación y distorsión.

Dependencias: numpy y matplotlib. Ejecución: python am_simulation.py
Los resultados se guardan en la carpeta `salidas` junto al programa.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# Parámetros reproducibles de la práctica.
FS = 20_000                 # Frecuencia de muestreo (muestras/s)
DURATION = 1.0              # Duración de la simulación (s)
FM = 5.0                    # Frecuencia de la señal de información (Hz)
FC = 500.0                  # Frecuencia de la portadora (Hz)
MU = 0.80                   # Índice de modulación, 0 <= mu <= 1
AC = 1.0                    # Amplitud de portadora (V, unidades relativas)
NOISE_SNR_DB = 10.0         # Relación señal/ruido añadida a la AM (dB)
ATTENUATION = 0.50          # Ganancia de canal con atenuación
CLIP_LEVEL = 0.65           # Umbral de recorte para el caso con distorsión
SEED = 2026                 # Semilla para resultados repetibles
OUT = Path(__file__).parent / "salidas"


def rms(x):
    """Calcula el valor eficaz de una señal."""
    return float(np.sqrt(np.mean(np.square(x))))


def snr_db(reference, observed):
    """Calcula SNR comparando una señal con su versión afectada."""
    error = observed - reference
    power_error = np.mean(error**2)
    if power_error == 0:
        return float("inf")
    return float(10 * np.log10(np.mean(reference**2) / power_error))


def single_sided_spectrum(x, fs):
    """Devuelve frecuencias y magnitud normalizada de la FFT unilateral."""
    n = len(x)
    spectrum = np.fft.rfft(x * np.hanning(n))
    magnitude = 2 * np.abs(spectrum) / np.sum(np.hanning(n))
    frequencies = np.fft.rfftfreq(n, d=1 / fs)
    return frequencies, magnitude


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    t = np.arange(0, DURATION, 1 / FS)

    # Señal de baja frecuencia normalizada que lleva la información.
    message = np.sin(2 * np.pi * FM * t)
    carrier = AC * np.cos(2 * np.pi * FC * t)

    # AM convencional (DSB-LC): la portadora conserva una envolvente positiva.
    am = AC * (1 + MU * message) * np.cos(2 * np.pi * FC * t)

    # Ruido blanco gaussiano escalado a la SNR indicada.
    noise_power = np.mean(am**2) / (10 ** (NOISE_SNR_DB / 10))
    noisy = am + rng.normal(0, np.sqrt(noise_power), size=t.size)

    # Canal con ganancia menor que uno: la forma ideal se conserva, cae la amplitud.
    attenuated = ATTENUATION * am

    # Distorsión no lineal: recorte duro que aplana los picos de la onda.
    distorted = np.clip(am, -CLIP_LEVEL, CLIP_LEVEL)

    # Gráficas temporales en una ventana corta para distinguir portadora y envolvente.
    view = t <= 0.012
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(t[view], message[view], color="#087f8c", lw=1.8)
    axes[0].set_title("Señal de información")
    axes[0].set_ylabel("Amplitud")
    axes[0].grid(alpha=.25)

    axes[1].plot(t[view], am[view], color="#17324d", lw=1.2, label="AM ideal")
    axes[1].plot(t[view], AC * (1 + MU * message[view]), color="#f28e2b", lw=1.6, label="Envolvente")
    axes[1].plot(t[view], -AC * (1 + MU * message[view]), color="#f28e2b", lw=1.0)
    axes[1].set_title("AM convencional en el tiempo")
    axes[1].set_ylabel("Amplitud")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=.25)

    axes[2].plot(t[view], am[view], color="#17324d", lw=1.1, label="AM ideal")
    axes[2].plot(t[view], noisy[view], color="#d1495b", lw=.7, alpha=.75, label="AM con ruido")
    axes[2].set_title(f"Canal con ruido blanco, SNR añadida = {NOISE_SNR_DB:.0f} dB")
    axes[2].set_xlabel("Tiempo (s)")
    axes[2].set_ylabel("Amplitud")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=.25)
    fig.savefig(OUT / "senales_tiempo.png", dpi=180)
    plt.close(fig)

    # Espectros: las bandas laterales aparecen a fc-fm y fc+fm.
    freq, mag = single_sided_spectrum(am, FS)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    band = (freq >= FC - 80) & (freq <= FC + 80)
    ax.plot(freq[band], mag[band], color="#087f8c", lw=1.7)
    ax.axvline(FC - FM, color="#f28e2b", ls="--", label=f"fc − fm = {FC-FM:.0f} Hz")
    ax.axvline(FC, color="#17324d", ls=":", label=f"Portadora fc = {FC:.0f} Hz")
    ax.axvline(FC + FM, color="#d1495b", ls="--", label=f"fc + fm = {FC+FM:.0f} Hz")
    ax.set_title("Espectro de AM alrededor de la portadora")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud (unidades relativas)")
    ax.legend()
    ax.grid(alpha=.25)
    fig.savefig(OUT / "espectro_am.png", dpi=180)
    plt.close(fig)

    # Comparación visual de efectos bajo condiciones controladas.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(t[view], am[view], label="Ideal", color="#17324d", lw=1.1)
    axes[0].plot(t[view], noisy[view], label="Con ruido", color="#d1495b", lw=.7)
    axes[0].set_title("Ruido: fluctuaciones aleatorias alrededor de la señal")
    axes[1].plot(t[view], am[view], label="Ideal", color="#17324d", lw=1.1)
    axes[1].plot(t[view], attenuated[view], label=f"Atenuada ×{ATTENUATION}", color="#087f8c", lw=1.1)
    axes[1].set_title("Atenuación: disminuye la amplitud")
    axes[2].plot(t[view], am[view], label="Ideal", color="#17324d", lw=1.1)
    axes[2].plot(t[view], distorted[view], label=f"Recorte a ±{CLIP_LEVEL}", color="#f28e2b", lw=1.1)
    axes[2].set_title("Distorsión: el recorte altera la envolvente y añade armónicos")
    for ax in axes:
        ax.set_ylabel("Amplitud")
        ax.grid(alpha=.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Tiempo (s)")
    fig.savefig(OUT / "efectos_canal.png", dpi=180)
    plt.close(fig)

    # Resumen de indicadores útil para comparar escenarios.
    rows = [
        ("AM ideal", 1.0, "sin ruido agregado", "sin distorsión"),
        ("Ruido blanco", 1.0, f"SNR objetivo {NOISE_SNR_DB:.1f} dB", "sin distorsión"),
        ("Atenuación", ATTENUATION, "sin ruido agregado", "forma conservada"),
        ("Recorte", 1.0, "sin ruido agregado", f"límite ±{CLIP_LEVEL:.2f}"),
    ]
    with (OUT / "resumen_escenarios.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Escenario", "Ganancia", "Condición", "Efecto observado"])
        writer.writerows(rows)

    print(f"Archivos generados en: {OUT}")
    print(f"SNR medida (ruido): {snr_db(am, noisy):.2f} dB")
    print(f"Ganancia de atenuación: {ATTENUATION:.2f} ({20*np.log10(ATTENUATION):.2f} dB)")
    print(f"RMS de AM ideal: {rms(am):.4f}")
    print(f"RMS después del recorte: {rms(distorted):.4f}")


if __name__ == "__main__":
    main()
