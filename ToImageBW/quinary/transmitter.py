import numpy as np
import sounddevice as sd
from PIL import Image
from settings import Settings

settings = Settings()
IMAGE_PATH = "example.png"

def image_to_bits(path):
    img = Image.open(path)
    img = img.convert("L")
    pixels = np.array(img)
    bits = (pixels > 128).astype(int)
    return bits.flatten()

def bits_to_base5(bits):
    padded_len = ((len(bits) + 2) // 3) * 3
    bits = np.pad(bits, (0, padded_len - len(bits)))

    symbols = []
    for i in range(0, len(bits), 3):
        value = bits[i]*4 + bits[i+1]*2 + bits[i+2]
        symbols.append(value % 5)
    return symbols

def generate_tone(freq, duration):
    t = np.linspace(0, duration, int(settings.ESTANDAR * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def symbol_to_freq(symbol):
    return [
        settings.FREQ_0,
        settings.FREQ_1,
        settings.FREQ_2,
        settings.FREQ_3,
        settings.FREQ_4
    ][symbol]

def symbols_to_signal(symbols):
    signal = []

    signal.extend(generate_tone(settings.FREQ_START, settings.DURACION_BIT))

    for s in symbols:
        signal.extend(generate_tone(symbol_to_freq(s), settings.DURACION_BIT))

    signal.extend(generate_tone(settings.FREQ_END, settings.DURACION_BIT))

    return np.array(signal)

bits = image_to_bits(IMAGE_PATH)
symbols = bits_to_base5(bits)

signal = symbols_to_signal(symbols)

print("Transmitiendo...")
sd.play(signal, settings.ESTANDAR)
sd.wait()
print("Fin")