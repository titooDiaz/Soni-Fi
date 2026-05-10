import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from settings import Settings
settings = Settings()

def get_frequency_strength(signal, target_freq):
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/settings.ESTANDAR)

    idx = np.argmin(np.abs(freqs - target_freq))

    return np.abs(fft[idx])

def is_tone(signal, target):
    freq = get_frequency_strength(signal, target)
    return abs(freq - target) < settings.TOLERANCIA

def record_until_end():
    print("Esperando inicio...")

    audio = []

    def callback(indata, frames, time, status):
        audio.extend(indata[:, 0])

    with sd.InputStream(callback=callback, samplerate=settings.ESTANDAR):
        started = False
        while True:
            if len(audio) > settings.ESTANDAR * 0.5:
                chunk = np.array(audio[-int(settings.ESTANDAR*0.5):])

                if not started and is_tone(chunk, settings.FREQ_START):
                    print("Inicio detectado")
                    audio.clear()
                    started = True

                elif started and is_tone(chunk, settings.FREQ_END):
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

        strength_1 = get_frequency_strength(chunk, settings.FREQ_1)
        strength_0 = get_frequency_strength(chunk, settings.FREQ_0)

        print(f"Bloque {i}: F1={strength_1:.2f}, F0={strength_0:.2f}")

        if strength_1 > strength_0 * 1.2:
            bits.append(1)
        elif strength_0 > strength_1 * 1.2:
            bits.append(0)
        else:
            print("Ruido")
            bits.append(1)

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