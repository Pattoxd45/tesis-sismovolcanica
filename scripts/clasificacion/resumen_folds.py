#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumen de la validacion cruzada (folds_cnn.py) por red y conjunto (recorte o ventana fija): F1 y
sensibilidad de cada clase en validacion (mejor epoca de cada fold) y en test (mejor modelo de cada fold),
con promedio, minimo, maximo y desviacion estandar entre los 5 folds; matriz de confusion de test
normalizada por fila (suma de los 5 folds); y cuantos sin caja de test se llaman volcanicos.

Uso: python resumen_folds.py <carpeta de folds_cnn.py>
     (deja resumen_folds.csv y matrices_test.csv en esa carpeta)
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import entrenar_cnn as ec


def cargar(carpeta):
    """Una fila por corrida de folds 1 a 5 con test: red, conjunto, fold y metricas de validacion y de test."""
    filas, matrices = [], {}
    for ruta in sorted(glob.glob(os.path.join(carpeta, "*.json"))):
        r = json.load(open(ruta))
        if "test" not in r or "configuracion" not in r:
            continue
        cfg, val, t = r["configuracion"], r["historia"][r["mejor_epoca"] - 1], r["test"]
        clave = (cfg["modelo"], cfg["conjunto"])
        for ev, m in (("validacion", val), ("test", t)):
            fila = dict(modelo=cfg["modelo"], conjunto=cfg["conjunto"], fold=cfg["fold"], evaluacion=ev,
                        f1_macro_volcanicas=m["f1_macro_volcanicas"],
                        sin_caja_volcanicos=sum(m["sin_caja_predichos"][k] for k in ec.VOLCANICAS))
            for k in ec.CLASES:
                fila[f"{k}_f1"], fila[f"{k}_sensibilidad"] = m[k]["f1"], m[k]["sensibilidad"]
            filas.append(fila)
        matrices.setdefault(clave, []).append(np.array(t["matriz"]))
    return pd.DataFrame(filas), matrices


def resumir(d):
    """Promedio, minimo, maximo y desviacion estandar (muestral, entre folds) de cada metrica."""
    met = [c for c in d.columns if c not in ("modelo", "conjunto", "fold", "evaluacion")]
    g = d.groupby(["modelo", "conjunto", "evaluacion"])[met]
    r = pd.concat({"promedio": g.mean(), "minimo": g.min(), "maximo": g.max(), "desviacion": g.std(ddof=1)},
                  names=["estadistico"]).reset_index()
    n = d.groupby(["modelo", "conjunto", "evaluacion"]).fold.nunique().rename("folds").reset_index()
    return r.merge(n, on=["modelo", "conjunto", "evaluacion"])


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    carpeta = sys.argv[1]
    d, matrices = cargar(carpeta)
    if d.empty:
        raise SystemExit(f"DETENIDO: no hay corridas con test en {carpeta}")
    r = resumir(d)
    r.to_csv(os.path.join(carpeta, "resumen_folds.csv"), index=False)
    filas = []
    for (modelo, conjunto), ms in sorted(matrices.items()):
        c = np.sum(ms, axis=0)
        # una clase sin eventos en test queda sin proporciones (NaN)
        p = c / np.where(c.sum(axis=1, keepdims=True) > 0, c.sum(axis=1, keepdims=True), np.nan)
        for k, fila in zip(ec.CLASES, p):
            filas.append(dict(modelo=modelo, conjunto=conjunto, folds=len(ms), clase_real=k,
                              **{f"pred_{j}": float(v) for j, v in zip(ec.CLASES, fila)}))
    mt = pd.DataFrame(filas)
    mt.to_csv(os.path.join(carpeta, "matrices_test.csv"), index=False)

    cols = ["f1_macro_volcanicas"] + [f"{k}_{x}" for k in ec.CLASES for x in ("f1", "sensibilidad")]
    for (modelo, conjunto, ev), g in r.groupby(["modelo", "conjunto", "evaluacion"], sort=True):
        t = g.set_index("estadistico").loc[["promedio", "minimo", "maximo", "desviacion"], cols].T
        print(f"\n{modelo}, {conjunto}, {ev} ({int(g.folds.iloc[0])} folds)")
        print(t.round(3).to_string())
    print("\nMatriz de confusion de test normalizada por fila (filas: clase real; columnas: prediccion):")
    for (modelo, conjunto), g in mt.groupby(["modelo", "conjunto"]):
        print(f"\n{modelo}, {conjunto} ({int(g.folds.iloc[0])} folds)")
        print(g.set_index("clase_real")[[f"pred_{k}" for k in ec.CLASES]].round(3).to_string())
    print(f"\nTablas en {os.path.join(carpeta, 'resumen_folds.csv')} y {os.path.join(carpeta, 'matrices_test.csv')}")


if __name__ == "__main__":
    main()
