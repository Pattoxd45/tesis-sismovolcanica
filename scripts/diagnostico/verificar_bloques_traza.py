"""
Detecta los cortes de la traza continua leyendo la fila 0 (tiempo Unix).

Uso: python verificar_bloques_traza.py
"""

import os
import numpy as np
from datetime import datetime, timezone

NPY = os.path.expanduser("~/tesis/datos/zenodo/NVCh_10h_continuous_trace.npy")
t = np.asarray(np.load(NPY, mmap_mode="r")[0])

# valor absoluto: el segundo corte retrocede en el tiempo
saltos = np.where(np.abs(np.diff(t) - 0.01) > 1.0)[0] + 1
print("cortes detectados en las muestras:", saltos)

for ini, fin in zip([0, *saltos], [*saltos, len(t)]):
    a = datetime.fromtimestamp(float(t[ini]), timezone.utc)
    b = datetime.fromtimestamp(float(t[fin - 1]), timezone.utc)
    dur = (fin - ini) / 100 / 3600
    print(f"[{ini:>9,} , {fin:>9,})  {a:%Y-%m-%d %H:%M:%S} -> {b:%Y-%m-%d %H:%M:%S} UTC  ({dur:.2f} h)")
