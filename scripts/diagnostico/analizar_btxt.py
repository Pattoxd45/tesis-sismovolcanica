import numpy as np
from scipy.signal import freqz

b = np.genfromtxt("/home/patto/tesis/ovdas-core/worker-deteccion/filters/b.txt")
b = np.atleast_1d(b).ravel()
print(f"Coeficientes: {b.size}")
print(f"Simetrico: {np.allclose(b, b[::-1])}  (fase lineal si es True)")
print(f"Suma (ganancia en DC): {b.sum():.4f}")

w, h = freqz(b, 1.0, worN=4096, fs=100.0)
mag = np.abs(h)
pico = w[int(np.argmax(mag))]
print(f"Ganancia maxima en {pico:.2f} Hz")

# banda de paso a -3 dB
lim = mag.max() / np.sqrt(2)
dentro = w[mag >= lim]
if dentro.size:
    print(f"Banda -3 dB: {dentro.min():.2f} - {dentro.max():.2f} Hz")

print("\nRespuesta (Hz -> ganancia normalizada):")
for f_ in [0, 0.5, 1, 2, 5, 10, 20, 30, 50]:
    i = int(np.argmin(np.abs(w - f_)))
    print(f"  {w[i]:>5.1f} Hz : {mag[i]/mag.max():.3f}")
