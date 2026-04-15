import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from settings import Settings

def detect_frequency(signal):
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/Settings.SAMPLE_RATE)

    idx = np.argmax(np.abs(fft))
    return abs(freqs[idx])

def is_tone(signal, target):
    freq = detect_frequency(signal)
    return abs(freq - target) < Settings.TOLERANCIA

def record_until_end():
    print("Esperando inicio...")

    audio = []

    def callback(indata, frames, time, status):
        audio.extend(indata[:, 0])

    with sd.InputStream(callback=callback, samplerate=Settings.SAMPLE_RATE):
        started = False
        while True:
            if len(audio) > Settings.SAMPLE_RATE * 0.5:
                chunk = np.array(audio[-int(Settings.SAMPLE_RATE*0.5):])

                if not started and is_tone(chunk, Settings.FREQ_START):
                    print("Inicio detectado")
                    audio.clear()
                    started = True

                elif started and is_tone(chunk, Settings.FREQ_END):
                    print("Fin detectado")
                    return np.array(audio)

def decode_audio(audio):
    samples_per_bit = int(Settings.SAMPLE_RATE * Settings.DURACION_BIT)
    bits = []

    total_bits = len(audio) // samples_per_bit

    print(f"bloques: {total_bits}")

    for i in range(total_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit

        chunk = audio[start:end]

        freq = detect_frequency(chunk)
        print(f"Bloque {i}: {int(freq)} Hz")

        if abs(freq - Settings.FREQ_1) < Settings.TOLERANCIA:
            bits.append(1)

        elif abs(freq - Settings.FREQ_0) < Settings.TOLERANCIA:
            bits.append(0)

        else:
            print("Ruido, ignorado")

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

WIDTH = 6
HEIGHT = 5

bits_to_image(bits, WIDTH, HEIGHT)