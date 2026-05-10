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

def decode_audio(audio):
    samples_per_bit = int(settings.ESTANDAR * settings.DURACION_BIT)
    bits = []

    total_bits = len(audio) // samples_per_bit

    print(f"bloques: {total_bits}")

    for i in range(total_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit

        chunk = audio[start:end]

        freq, _, _ = detect_frequency(chunk)
        print(f"Bloque {i}: {int(freq) if freq else 0} Hz")

        if freq == settings.FREQ_1:
            bits.append(1)

        elif freq == settings.FREQ_0:
            bits.append(0)

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

bits = decode_audio(audio)

print("Bits:", bits)

WIDTH = settings.WIDTH
HEIGHT = settings.HEIGHT

bits_to_image(bits, WIDTH, HEIGHT)