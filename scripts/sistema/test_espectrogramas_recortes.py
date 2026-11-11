#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para espectrogramas_recortes.py: tamano y orientacion del espectrograma, columnas de
cada recorte y etiqueta propuesta segun la caja de Jose que le toca.

Uso: python test_espectrogramas_recortes.py
"""

import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import espectrogramas_recortes as er

FS = er.FS


def caso(nombre, ok):
    print(f"  {'OK   ' if ok else 'FALLA'} {nombre}")
    return ok


def main():
    todo = True
    sos = butter(er.ORDEN_FILTRO, list(er.BANDA_FILTRO), btype="bandpass", fs=FS, output="sos")
    t = np.arange(30000) / FS
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 5.0 * t) * 100 + rng.normal(0, 1, t.size)
    db, vmin, vmax = er.espectrograma(x, sos)
    fila = int(np.argmax(db.mean(axis=1)))
    todo &= caso("300 s dan 77 filas (0-15 Hz) y 465 columnas, como las imagenes de Jose", db.shape == (77, 465))
    todo &= caso("un tono de 5 Hz queda en la fila de 5 Hz y la escala cubre 60 dB",
                 abs(fila * FS / er.NFFT - 5.0) < 0.2 and abs(vmax - vmin - 60) < 1e-9 and vmax == db.max())

    plano = np.full_like(db, vmin)
    plano[fila] = vmax
    dbs = {c: plano for c in ("Z", "N", "E")}
    lims = {c: (vmin, vmax) for c in dbs}
    img = er.imagen(dbs, lims, np.arange(10, 30))
    distintas = [np.flatnonzero((img[i * 77:(i + 1) * 77] != img[i * 77]).any(axis=(1, 2))).tolist() for i in range(3)]
    todo &= caso("imagen: Z, N y E apiladas, 20 columnas, frecuencias bajas abajo",
                 img.shape == (231, 20, 3) and img.dtype == np.uint8 and distintas == [[76 - fila]] * 3)

    ruta = os.path.join(tempfile.mkdtemp(), "jose.png")
    plt.imsave(ruta, db, cmap="jet", vmin=vmin, vmax=vmax, origin="lower")
    jose = (plt.imread(ruta)[:, :, :3] * 255).round().astype(np.uint8)
    todo &= caso("con todas las columnas, la imagen de Z es la que guarda plt.imsave de Jose, pixel a pixel",
                 np.array_equal(er.imagen({c: db for c in ("Z", "N", "E")}, {c: (vmin, vmax) for c in ("Z", "N", "E")},
                                          np.arange(db.shape[1]))[:77], jose))

    todo &= caso("columnas cuyo centro cae en el recorte", er.columnas(0, 1000, 465).tolist() == list(range(14)))
    todo &= caso("recorte mas corto que el salto: la columna mas cercana", er.columnas(300, 310, 465).tolist() == [3])

    cajas = pd.DataFrame(dict(ini=[1000, 5000], fin=[2000, 6000], clase=["VT", "LP"], tipo_marca=["PRINCIPAL", "EXTRA"]))
    et = er.etiquetar(cajas, [(1100, 2100), (5900, 9000), (20000, 21000), (1500, 1600)])
    todo &= caso("etiqueta: clase de la caja con IoU >= 0,3; 'fuera' si toca con menos o la caja ya tiene recorte; "
                 "'sin caja' si no toca nada",
                 [r["etiqueta"] for r in et] == ["VT", "fuera", "sin caja", "fuera"]
                 and abs(et[0]["iou"] - 900 / 1100) < 1e-9 and et[0]["tipo_caja"] == "PRINCIPAL"
                 and et[1]["clase_caja"] == "LP" and et[3]["clase_caja"] == "")

    print("TODOS PASAN" if todo else "HAY FALLAS")
    sys.exit(0 if todo else 1)


if __name__ == "__main__":
    main()
