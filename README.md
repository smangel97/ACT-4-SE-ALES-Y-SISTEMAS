# Simulación de modulación en amplitud (AM)

Este proyecto implementa AM convencional de doble banda lateral con portadora (DSB-LC) y compara la señal ideal con tres condiciones de canal: ruido blanco, atenuación y distorsión por recorte.

## Requisitos

- Python 3.9 o posterior
- NumPy
- Matplotlib

## Instalación y ejecución

```bash
python -m pip install -r requirements.txt
python am_simulation.py
```

El programa crea `salidas/` con tres imágenes (`senales_tiempo.png`, `espectro_am.png`, `efectos_canal.png`) y una tabla (`resumen_escenarios.csv`). La semilla aleatoria fija permite repetir el caso de ruido.

## Modelo

La señal de información es `m(t) = sin(2π fm t)` y la portadora es `c(t) = Ac cos(2π fc t)`. El programa usa:

`s(t) = Ac [1 + μ m(t)] cos(2π fc t)`

Parámetros iniciales: `fs = 20 kHz`, `fm = 5 Hz`, `fc = 500 Hz`, `μ = 0.80` y `Ac = 1`. Como `μ ≤ 1`, la envolvente ideal no se invierte y puede seguir la forma del mensaje.

- **Ruido:** ruido blanco gaussiano agregado para una SNR objetivo de 10 dB.
- **Atenuación:** ganancia de canal de 0.50, equivalente a aproximadamente −6.02 dB en amplitud.
- **Distorsión:** recorte duro de la señal en ±0.65, que aplana partes de la onda y genera componentes armónicas.

La FFT usa ventana Hann para reducir la fuga espectral. En el espectro AM aparecen componentes en `fc` y bandas laterales alrededor de `fc ± fm`.

## Estructura

```text
am_amplitude_modulation/
├── am_simulation.py
├── README.md
├── requirements.txt
└── salidas/                 # se genera al ejecutar el programa
```

## Referencias

- Haykin, S. (2001). *Communication Systems* (4.ª ed.). John Wiley & Sons.
- Lathi, B. P., & Ding, Z. (2010). *Modern Digital and Analog Communication Systems* (4.ª ed.). Oxford University Press.
