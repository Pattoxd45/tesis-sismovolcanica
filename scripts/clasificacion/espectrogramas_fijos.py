#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segunda forma de armar las imagenes, pedida por el profesor: cada recorte con una duracion fija. Desde
el inicio del recorte se toman D segundos del espectrograma de la ventana (parametros de Jose): si el
evento dura mas se corta, si dura menos se prolonga con la senal que sigue; si no alcanza antes del fin
de la ventana, se toman sus ultimos D segundos. Usa los mismos recortes y etiquetas de
espectrogramas_recortes.py; solo cambia la imagen.

Uso: python espectrogramas_fijos.py --duracion 60
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import butter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "sistema"))
import espectrogramas_recortes as er

ej, lib = er.ej, er.lib
FS = er.FS
VENTANA_S = ej.N_VENTANA / FS
INDICE = os.path.join(er.rv.DIR_SALIDA, "espectrogramas_recortes.csv")


def tramo_fijo(ini_s, duracion_s, ventana_s=VENTANA_S):
    """Inicio y fin (s) del tramo de duracion fija: desde el inicio del recorte, sin salir de la ventana."""
    a = max(min(ini_s, ventana_s - duracion_s), 0.0)
    return a, min(a + duracion_s, ventana_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duracion", type=float, required=True, help="duracion fija en segundos")
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N ventanas, para probar")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")

    d = args.duracion
    nombre = f"espectrogramas_fijos_{d:g}s"
    dir_img = os.path.join(lib.DIR_TESIS, "datos", nombre)
    os.makedirs(dir_img, exist_ok=True)
    idx = pd.read_csv(INDICE)
    ventanas = list(dict.fromkeys(idx.ventana))
    if args.limite:
        ventanas = ventanas[:args.limite]
        idx = idx[idx.ventana.isin(ventanas)]
    sos = butter(er.ORDEN_FILTRO, list(er.BANDA_FILTRO), btype="bandpass", fs=FS, output="sos")
    print(f"{len(idx)} recortes de {len(ventanas)} ventanas; duracion fija {d:g} s desde el inicio de cada recorte")

    ini_f, fin_f, ncol, t0 = {}, {}, {}, time.time()
    for n, (v, g) in enumerate(idx.groupby("ventana", sort=False), 1):
        w = np.load(os.path.join(ej.DIR_SALIDA, f"{v}_zne.npz"))
        esp = {c: er.espectrograma(w[c], sos) for c in ("Z", "N", "E")}
        dbs = {c: x[0] for c, x in esp.items()}
        lims = {c: (x[1], x[2]) for c, x in esp.items()}
        for r in g.itertuples():
            a, b = tramo_fijo(r.ini_s, d)
            col = er.columnas(a * FS, b * FS, dbs["Z"].shape[1])
            Image.fromarray(er.imagen(dbs, lims, col)).save(os.path.join(dir_img, r.archivo))
            ini_f[r.Index], fin_f[r.Index], ncol[r.Index] = a, b, col.size
        if n % 1000 == 0:
            print(f"  {n}/{len(ventanas)} ventanas en {(time.time() - t0) / 60:.1f} min", flush=True)

    idx = idx.assign(ini_fijo_s=pd.Series(ini_f), fin_fijo_s=pd.Series(fin_f), columnas=pd.Series(ncol))
    ruta_idx = os.path.join(er.rv.DIR_SALIDA, f"{nombre}.csv")
    idx.to_csv(ruta_idx, index=False)
    corridos = int((idx.ini_fijo_s < idx.ini_s - 1e-9).sum())
    print(f"\nTiempo: {(time.time() - t0) / 60:.1f} min; columnas por imagen: {idx.columnas.min()} a {idx.columnas.max()}")
    print(f"Recortes que se cortan (duran mas de {d:g} s): {int((idx.dur_s > d).sum())}; que se prolongan: "
          f"{int((idx.dur_s < d).sum())}; que empiezan antes porque no alcanzaban dentro de la ventana: {corridos}")
    ruta_m = os.path.join(er.rv.DIR_SALIDA, f"muestra_{nombre}.png")
    er.muestra(idx, dir_img, ruta_m)
    print(f"Imagenes en {dir_img}; indice en {ruta_idx}; hoja de muestra en {ruta_m}")


if __name__ == "__main__":
    main()
