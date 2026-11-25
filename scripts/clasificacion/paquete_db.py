#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arma el paquete en dB, para variar el rango de dB durante el entrenamiento: por ventana, el espectrograma de
Z, N y E en dB (0-15 Hz, parametros de Jose) y el maximo de cada componente; un indice con la etiqueta, la
particion y el fold de Jose y las columnas de cada recorte, del recorte variable y del tramo de duracion
fija. Compara las imagenes que salen del paquete con los PNG de entrenamiento en las primeras ventanas.

Uso: python paquete_db.py [--duracion-fija 60] [--verificar 20] [--limite N]
     (deja ~/Descargas/clasificacion_db.zip)
"""

import argparse
import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import butter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "sistema"))
import entrenar_cnn as ec
import espectrogramas_recortes as er
import paquete_colab as pc

SALIDA = os.path.expanduser("~/Descargas/clasificacion_db.zip")


def rango_columnas(ini_s, fin_s, n_col):
    """Primera y ultima + 1 de las columnas del recorte, las mismas que usa espectrogramas_recortes.py."""
    k = er.columnas(ini_s * er.FS, fin_s * er.FS, n_col)
    if not np.array_equal(k, np.arange(k[0], k[-1] + 1)):
        raise SystemExit(f"DETENIDO: columnas no contiguas para el tramo {ini_s}-{fin_s} s")
    return int(k[0]), int(k[-1]) + 1


def diferencias(a, b):
    """Pixeles distintos (en algun canal) y mayor diferencia entre dos imagenes del mismo tamano."""
    if a.shape != b.shape:
        return a.size, 255
    d = np.abs(a.astype(int) - b.astype(int))
    return int((d.max(axis=2) > 0).sum()), int(d.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duracion-fija", type=float, default=60.0, help="tramo de duracion fija ya generado (s)")
    ap.add_argument("--verificar", type=int, default=20, help="ventanas en que se comparan las imagenes con los PNG")
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N ventanas, para probar")
    ap.add_argument("--indice", default=os.path.join(er.rv.DIR_SALIDA, "espectrogramas_recortes.csv"))
    ap.add_argument("--indice-fijo", default=None, help="por defecto, espectrogramas_fijos_<D>s.csv junto al indice")
    ap.add_argument("--ventanas", default=er.ej.DIR_SALIDA, help="carpeta con <ventana>_zne.npz")
    ap.add_argument("--imagenes", default=er.DIR_IMAGENES, help="PNG del recorte variable, para comparar")
    ap.add_argument("--imagenes-fijo", default=None, help="PNG del tramo fijo, para comparar")
    ap.add_argument("--splits", default=pc.DIR_SPLITS)
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()

    d = args.duracion_fija
    sufijo = f"fijo{d:g}s"
    ruta_fijo = args.indice_fijo or os.path.join(os.path.dirname(args.indice), f"espectrogramas_fijos_{d:g}s.csv")
    dir_fijo = args.imagenes_fijo or os.path.join(er.lib.DIR_TESIS, "datos", f"espectrogramas_fijos_{d:g}s")
    idx = pc.indice_con_folds(pd.read_csv(os.path.expanduser(args.indice)), pc.folds_jose(args.splits))
    fijo = pd.read_csv(os.path.expanduser(ruta_fijo))[["archivo", "ini_fijo_s", "fin_fijo_s"]]
    idx = idx.merge(fijo, on="archivo", how="left", validate="one_to_one")
    if idx.ini_fijo_s.isna().any():
        raise SystemExit(f"DETENIDO: {int(idx.ini_fijo_s.isna().sum())} recortes sin tramo fijo en {ruta_fijo}")
    ventanas = list(dict.fromkeys(idx.ventana))
    if args.limite:
        ventanas = ventanas[:args.limite]
        idx = idx[idx.ventana.isin(ventanas)]
    sos = butter(er.ORDEN_FILTRO, list(er.BANDA_FILTRO), btype="bandpass", fs=er.FS, output="sos")
    raiz = pc.raiz(args.salida)
    print(f"{len(idx)} recortes de {len(ventanas)} ventanas; tramo fijo de {d:g} s; comparacion con los PNG en "
          f"{min(args.verificar, len(ventanas))} ventanas")

    cols = {k: {} for k in ("col_ini", "col_fin", f"col_ini_{sufijo}", f"col_fin_{sufijo}")}
    comparados, distintos, pixeles, peor, t0 = 0, 0, 0, 0, time.time()
    salida = os.path.expanduser(args.salida)
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    # float32: con float16 la escala de colores cambiaria en algunos pixeles respecto de los PNG
    with zipfile.ZipFile(salida, "w", compression=zipfile.ZIP_STORED) as z:
        for n, (v, g) in enumerate(idx.groupby("ventana", sort=False), 1):
            w = np.load(os.path.join(args.ventanas, f"{v}_zne.npz"))
            esp = {c: er.espectrograma(w[c], sos) for c in ec.COMPONENTES}
            dbs = {c: x[0].astype(np.float32) for c, x in esp.items()}
            n_col = dbs["Z"].shape[1]
            buf = io.BytesIO()
            np.savez(buf, **dbs, **{f"vmax_{c}": np.float64(x[2]) for c, x in esp.items()})
            z.writestr(f"{raiz}/ventanas/{v}.npz", buf.getvalue())
            lims = {c: (x[1], x[2]) for c, x in esp.items()}
            for r in g.itertuples():
                a, b = rango_columnas(r.ini_s, r.fin_s, n_col)
                af, bf = rango_columnas(r.ini_fijo_s, r.fin_fijo_s, n_col)
                cols["col_ini"][r.Index], cols["col_fin"][r.Index] = a, b
                cols[f"col_ini_{sufijo}"][r.Index], cols[f"col_fin_{sufijo}"][r.Index] = af, bf
                if n <= args.verificar:
                    for (c0, c1), dir_png in (((a, b), args.imagenes), ((af, bf), dir_fijo)):
                        ruta = os.path.join(dir_png, r.archivo)
                        if os.path.exists(ruta):
                            k, m = diferencias(ec.colorear(dbs, lims, slice(c0, c1)), np.asarray(Image.open(ruta).convert("RGB")))
                            comparados += 1
                            distintos += k > 0
                            pixeles += k
                            peor = max(peor, m)
            if n % 500 == 0:
                print(f"  {n}/{len(ventanas)} ventanas en {(time.time() - t0) / 60:.1f} min", flush=True)
        idx = idx.assign(**{k: pd.Series(v) for k, v in cols.items()}, duracion_fija_s=d)
        z.writestr(f"{raiz}/indice.csv", idx.to_csv(index=False))
        z.write(os.path.join(AQUI, "entrenar_cnn.py"), f"{raiz}/entrenar_cnn.py")

    print(f"\nTiempo: {(time.time() - t0) / 60:.1f} min")
    print(pd.crosstab(idx.etiqueta, idx.fold.astype("object").fillna("test"), margins=True, margins_name="total").to_string())
    if comparados:
        print(f"\nComparacion con los PNG: {comparados} imagenes; {distintos} con algun pixel distinto "
              f"({pixeles} pixeles en total); mayor diferencia en un canal: {peor} de 255")
    else:
        print("\nNo se encontraron los PNG para comparar")
    print(f"Paquete en {salida} ({os.path.getsize(salida) / 1e9:.2f} GB); al descomprimirlo queda la carpeta {raiz}/")


if __name__ == "__main__":
    main()
