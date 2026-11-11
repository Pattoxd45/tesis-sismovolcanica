#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arma el paquete para entrenar en Colab: las imagenes de los recortes, un indice con la etiqueta
propuesta, la particion de Jose (desarrollo o test) y su fold de validacion cruzada por ventana, y el
script de entrenamiento.

Uso: python paquete_colab.py   (deja ~/Descargas/clasificacion_colab.zip)
"""

import os
import sys
import zipfile

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_TESIS = os.path.expanduser("~/tesis")
INDICE = os.path.join(DIR_TESIS, "resultados", "sistema", "espectrogramas_recortes.csv")
DIR_IMAGENES = os.path.join(DIR_TESIS, "datos", "espectrogramas_recortes")
DIR_SPLITS = os.path.join(DIR_TESIS, "datos", "jose", "Pato", "splits")
SALIDA = os.path.expanduser("~/Descargas/clasificacion_colab.zip")
RAIZ = "clasificacion_colab"
N_FOLDS = 5


def nombre(ruta):
    return os.path.basename(ruta.strip()).replace("_300s.png", "")


def folds_jose(dir_splits):
    """Fold de validacion de cada ventana de desarrollo, segun fold_0k_val.txt de Jose."""
    fold = {}
    for k in range(1, N_FOLDS + 1):
        with open(os.path.join(dir_splits, f"fold_{k:02d}_val.txt")) as f:
            for linea in f:
                if linea.strip():
                    v = nombre(linea)
                    if v in fold:
                        raise SystemExit(f"DETENIDO: {v} esta en la validacion de los folds {fold[v]} y {k}")
                    fold[v] = k
    with open(os.path.join(dir_splits, "desarrollo.txt")) as f:
        desarrollo = {nombre(l) for l in f if l.strip()}
    if set(fold) != desarrollo:
        raise SystemExit(f"DETENIDO: los folds no cubren exactamente desarrollo.txt ({len(fold)} y {len(desarrollo)})")
    return fold


def indice_con_folds(idx, fold):
    idx = idx.copy()
    idx["fold"] = idx.ventana.map(fold).astype("Int64")
    malas = idx[((idx.split == "desarrollo") & idx.fold.isna()) | ((idx.split == "test") & idx.fold.notna())]
    if len(malas):
        raise SystemExit(f"DETENIDO: {malas.ventana.nunique()} ventanas con particion y fold que no calzan")
    return idx


def main():
    idx = indice_con_folds(pd.read_csv(INDICE), folds_jose(DIR_SPLITS))
    faltan = [a for a in idx.archivo if not os.path.exists(os.path.join(DIR_IMAGENES, a))]
    if faltan:
        raise SystemExit(f"DETENIDO: faltan {len(faltan)} imagenes, por ejemplo {faltan[0]}")
    print(f"{len(idx)} recortes de {idx.ventana.nunique()} ventanas")
    print(pd.crosstab(idx.etiqueta, idx.fold.astype("object").fillna("test"), margins=True, margins_name="total").to_string())

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    # PNG ya viene comprimido: se guarda tal cual
    with zipfile.ZipFile(SALIDA, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(f"{RAIZ}/indice.csv", idx.to_csv(index=False))
        z.write(os.path.join(AQUI, "entrenar_cnn.py"), f"{RAIZ}/entrenar_cnn.py")
        for a in idx.archivo:
            z.write(os.path.join(DIR_IMAGENES, a), f"{RAIZ}/imagenes/{a}")
    print(f"\nPaquete en {SALIDA} ({os.path.getsize(SALIDA) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
