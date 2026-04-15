import numpy as np
import sounddevice as sd
from PIL import Image
from settings import Settings

settings = Settings()
IMAGE_PATH = "example.png"

def image_to_bits(path):
    img = Image.open(path)
    width, height = img.size
    img = img.convert("L")
    pixels = np.array(img)
    bits = (pixels > 128).astype(int)
    return bits.flatten(), width, height

def generate_tone(freq, duration):
    t = np.linspace(0, duration, int(settings.ESTANDAR * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def bits_to_signal(bits):
    signal = []
    
    signal.extend(generate_tone(settings.FREQ_START, settings.DURACION_BIT))

    for bit in bits:
        if bit == 0:
            tone = generate_tone(settings.FREQ_0, settings.DURACION_BIT)
        else:
            tone = generate_tone(settings.FREQ_1, settings.DURACION_BIT)
        signal.extend(tone)

    signal.extend(generate_tone(settings.FREQ_END, settings.DURACION_BIT))

    return np.array(signal)


bits, w, h = image_to_bits(IMAGE_PATH)

signal = bits_to_signal(bits)

print("Transmitiendo...")
sd.play(signal, settings.ESTANDAR)
sd.wait()
print("Fin")