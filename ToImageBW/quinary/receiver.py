import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from settings import Settings

settings = Settings()

def detect_frequency(signal):
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/settings.ESTANDAR)
    magnitudes = np.abs(fft)

    targets = [
        settings.FREQ_0,
        settings.FREQ_1,
        settings.FREQ_2,
        settings.FREQ_3,
        settings.FREQ_4,
        settings.FREQ_START,
        settings.FREQ_END
    ]

    powers = {}

    for target in targets:
        idx = np.where((freqs > target - settings.TOLERANCIA) & 
                       (freqs < target + settings.TOLERANCIA))[0]

        if len(idx) == 0:
            powers[target] = 0
        else:
            powers[target] = np.max(magnitudes[idx])

    best_freq = max(powers, key=powers.get)
    best_power = powers[best_freq]

    return best_freq, best_power, powers

def is_dominant_tone(signal, target):
    freq, power, powers = detect_frequency(signal)

    if freq != target:
        return False

    sorted_powers = sorted(powers.values(), reverse=True)

    if len(sorted_powers) < 2:
        return False

    second_power = sorted_powers[1]

    return (
        power > settings.UMBRAL_MINIMO and
        power > second_power * 2
    )

def record_until_end():
    print("Esperando inicio...")

    audio = []

    def callback(indata, frames, time, status):
        audio.extend(indata[:, 0])

    with sd.InputStream(callback=callback, samplerate=settings.ESTANDAR):
        started = False
        while True:
            if len(audio) > settings.ESTANDAR * 0.1:
                chunk = np.array(audio[-int(settings.ESTANDAR*0.1):])

                if not started and is_dominant_tone(chunk, settings.FREQ_START):
                    print("Inicio detectado")
                    audio.clear()
                    started = True

                elif started and is_dominant_tone(chunk, settings.FREQ_END):
                    print("Fin detectado")
                    return np.array(audio)

def freq_to_symbol(freq):
    mapping = {
        settings.FREQ_0: 0,
        settings.FREQ_1: 1,
        settings.FREQ_2: 2,
        settings.FREQ_3: 3,
        settings.FREQ_4: 4
    }
    return mapping.get(freq, None)

def decode_audio(audio):
    samples_per_symbol = int(settings.ESTANDAR * settings.DURACION_BIT)
    symbols = []

    total = len(audio) // samples_per_symbol

    print(f"bloques: {total}")

    for i in range(total):
        start = i * samples_per_symbol
        end = start + samples_per_symbol

        chunk = audio[start:end]

        freq, _, _ = detect_frequency(chunk)
        print(f"Bloque {i}: {int(freq) if freq else 0} Hz")

        symbol = freq_to_symbol(freq)
        if symbol is not None:
            symbols.append(symbol)

    return symbols

def base5_to_bits(symbols):
    bits = []

    for s in symbols:
        b2 = (s >> 2) & 1
        b1 = (s >> 1) & 1
        b0 = s & 1
        bits.extend([b2, b1, b0])

    return bits

def bits_to_image(bits, width, height):
    if len(bits) < width * height:
        print("No hay suficientes bits")
        return

    img = np.array(bits[:width*height])
    img = img.reshape((height, width))

    plt.imshow(img, cmap='gray')
    plt.title("Imagen recibida")
    plt.show()

audio = record_until_end()

symbols = decode_audio(audio)

bits = base5_to_bits(symbols)

print("Bits:", bits)

bits_to_image(bits, settings.WIDTH, settings.HEIGHT)