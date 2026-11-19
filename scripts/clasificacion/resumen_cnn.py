#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumen de los resultados del clasificador (JSON de entrenar_cnn.py) con un protocolo: por red y forma
de imagen, cada fold, el promedio y la desviacion de los folds, y el resultado en test, sobre los eventos
de Jose; aparte, la parte de los recortes sin caja que la red toma por volcanicos. Los recortes que la
duracion minima deja fuera cuentan como rechazados (asi trabaja el sistema completo).

Uso: python resumen_cnn.py <carpeta de resultados> [--protocolo _pesos_completo_lr0.0001_min5s_sin_falsas]
"""

import argparse
import glob
import json
import os

import pandas as pd

CLASES = ["VT", "LP", "TR", "rechazo"]
PROTOCOLO = "_pesos_completo_lr0.0001_min5s_sin_falsas"


def leer(ruta):
    """Metricas de la epoca informada, sumando los recortes cortos como rechazados."""
    r = json.load(open(ruta))
    cfg, m = r["configuracion"], r["historia"][r["mejor_epoca"] - 1]
    c, cortos = m["matriz"], cfg.get("cortos_fuera_de_validacion") or {}
    fila = {k: sum(c[i]) + cortos.get(k, 0) for i, k in enumerate(CLASES)}
    out = dict(modelo=cfg["modelo"], conjunto=cfg.get("conjunto", "recorte"), fold=cfg["fold"],
               evaluacion=cfg.get("evaluacion", "validacion"), epoca=r["mejor_epoca"], f1=m["f1_macro_volcanicas"])
    out.update({f"sens_{k}": c[i][i] / fila[k] for i, k in enumerate(CLASES[:3])})
    out["volcanicos"] = sum(c[i][i] for i in range(3)) / sum(fila[k] for k in CLASES[:3])
    out["falsos"] = sum(c[3][:3]) / fila["rechazo"]
    sc = m.get("sin_caja_predichos")
    n_sc = sum(sc.values()) + cfg.get("cortos_sin_caja", 0) if sc else 0
    out["sin_caja_volcanico"] = sum(sc[k] for k in CLASES[:3]) / n_sc if n_sc else float("nan")
    out["n_volcanicos"] = sum(fila[k] for k in CLASES[:3])
    out["n_rechazo"] = fila["rechazo"]
    out["s_epoca"] = sum(h["segundos"] for h in r["historia"]) / len(r["historia"])
    return out


def tabla(carpeta, protocolo=PROTOCOLO):
    rutas = sorted(glob.glob(os.path.join(carpeta, f"*{protocolo}.json")))
    return pd.DataFrame([leer(r) for r in rutas])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--protocolo", default=PROTOCOLO)
    args = ap.parse_args()
    d = tabla(os.path.expanduser(args.carpeta), args.protocolo)
    if d.empty:
        raise SystemExit(f"No hay resultados con el protocolo {args.protocolo} en {args.carpeta}")
    cols = ["f1", "sens_VT", "sens_LP", "sens_TR", "volcanicos", "falsos", "sin_caja_volcanico"]
    print("Columnas (eventos de Jose): F1 macro de VT, LP y TR; sensibilidad de VT, LP y TR; volcanicos bien clasificados; "
          "parte de AV e IC tomada por volcanico. Ultima columna (sistema completo): parte de los recortes sin caja tomada "
          "por volcanico. Los recortes cortos cuentan como rechazados.\n")
    for (modelo, conjunto), g in d.groupby(["modelo", "conjunto"]):
        print(f"{modelo} ({conjunto})")
        cv = g[g.evaluacion == "validacion"].sort_values("fold")
        for r in cv.itertuples():
            print(f"  fold {r.fold} (epoca {r.epoca}): " + "  ".join(f"{getattr(r, k):.3f}" for k in cols)
                  + f"  ({r.s_epoca:.0f} s por epoca)")
        if len(cv) > 1:
            print(f"  promedio de {len(cv)} folds:  " + "  ".join(f"{cv[k].mean():.3f}" for k in cols))
            print(f"  desviacion:            " + "  ".join(f"{cv[k].std():.3f}" for k in cols))
        for r in g[g.evaluacion == "test"].itertuples():
            print(f"  TEST ({r.n_volcanicos} volcanicos, {r.n_rechazo} de rechazo; {r.epoca} epocas): "
                  + "  ".join(f"{getattr(r, k):.3f}" for k in cols))
        print()
    ruta = os.path.join(os.path.expanduser(args.carpeta), f"resumen{args.protocolo}.csv")
    d.to_csv(ruta, index=False)
    print(f"Tabla en {ruta}")


if __name__ == "__main__":
    main()
