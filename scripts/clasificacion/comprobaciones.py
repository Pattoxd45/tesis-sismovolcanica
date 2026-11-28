#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprobaciones sobre los resultados del clasificador, para entender por que la traza queda bajo test:
1) traslape en el tiempo entre las ventanas de Jose y la traza continua (la traza es solo para evaluar);
2) en test, sensibilidad de los recortes que calzan con la caja principal (al centro de la ventana) y con
   las cajas extra (descentradas), y por IoU del recorte;
3) en la traza, intervalo de confianza de Wilson (95 %) de la sensibilidad de cada clase, con el
   promedio de test al lado, e IoU de los recortes de la traza y de test;
4) cuantos sin caja de test clasifica cada red como volcanicos.
Ademas dibuja las matrices de confusion normalizadas por fila de test y de la traza.

Uso: python comprobaciones.py   (deja comprobaciones.txt y las figuras en resultados/clasificacion)
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "sistema"))
import entrenar_cnn as ec
import espectrogramas_recortes as er

lib = er.rv.lib
RES = os.path.join(lib.DIR_TESIS, "resultados")
FOLDS = os.path.join(RES, "clasificacion", "folds")
SALIDA = os.path.join(RES, "clasificacion")
COBERTURA = os.path.join(RES, "sistema", "cobertura_jose.csv")
INDICE = os.path.join(er.rv.DIR_SALIDA, "espectrogramas_recortes.csv")
TRAZA = os.path.join(RES, "sistema", "sistema_traza_recortes.csv")
MODELOS_TRAZA = os.path.join(RES, "sistema", "sistema_traza_modelos.csv")
# J2000 (2000-01-01 12:00 UTC) en segundos Unix, como en winston_lib.py
J2K_A_UNIX = 946728000.0
REDES = ["efficientnet_b0", "resnet50", "vgg16"]
NOMBRE = {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet50", "vgg16": "VGG16"}


def wilson(k, n, z=1.96):
    """Intervalo de confianza de Wilson para una proporcion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - m), float(c + m)


def traslapes(ventanas, tramos):
    """Ventanas (inicio y fin Unix) que comparten tiempo con algun tramo de la traza."""
    t = np.zeros(len(ventanas), dtype=bool)
    for a, b in tramos:
        t |= (ventanas.ini_unix.to_numpy() < b) & (ventanas.fin_unix.to_numpy() > a)
    return t


def predicciones_test(carpeta):
    """Predicciones de test de los 5 folds, con la red, el conjunto y el fold."""
    filas = []
    for ruta in sorted(glob.glob(os.path.join(carpeta, "*_test_predicciones.csv"))):
        js = ruta.replace("_test_predicciones.csv", ".json")
        if not os.path.exists(js):
            continue
        cfg = json.load(open(js))["configuracion"]
        filas.append(pd.read_csv(ruta).assign(modelo=cfg["modelo"], conjunto=cfg["conjunto"], fold=cfg["fold"]))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


def sensibilidad(d, grupos):
    """Sensibilidad por clase (promedio entre folds) para cada grupo."""
    d = d.assign(bien=d.clase == d.prediccion)
    s = d.groupby(grupos + ["fold", "clase"]).bien.mean().groupby(grupos + ["clase"]).mean().unstack("clase")
    n = d[d.fold == d.fold.min()].groupby(grupos + ["clase"]).size().unstack("clase")
    return s, n


def matrices(plt, mats, titulo, ruta):
    """Una fila por conjunto y una columna por red, con la proporcion de cada celda."""
    conjuntos = sorted({c for _, c in mats})
    fig, ax = plt.subplots(len(conjuntos), len(REDES), figsize=(4.2 * len(REDES), 3.9 * len(conjuntos)), squeeze=False)
    nombres = ["VT", "LP", "TR", "Rechazo"]
    for i, conj in enumerate(conjuntos):
        for j, red in enumerate(REDES):
            a = ax[i, j]
            c = mats.get((red, conj))
            if c is None:
                a.axis("off")
                continue
            p = c / np.where(c.sum(axis=1, keepdims=True) > 0, c.sum(axis=1, keepdims=True), np.nan)
            a.imshow(p, cmap="Blues", vmin=0, vmax=1)
            for r in range(4):
                for k in range(4):
                    a.text(k, r, f"{p[r, k]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9,
                           color="white" if p[r, k] > 0.6 else "black")
            a.set_xticks(range(4), nombres, fontsize=8)
            a.set_yticks(range(4), nombres, fontsize=8)
            a.set_xlabel("Clase predicha", fontsize=8)
            a.set_ylabel("Clase real", fontsize=8)
            a.set_title(f"{NOMBRE[red]}, {'recorte' if conj == 'recorte' else 'ventana fija de 60 s'}", fontsize=9)
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []

    def p(x=""):
        print(x)
        out.append(str(x))

    p("1. TRASLAPE ENTRE LAS VENTANAS DE JOSE Y LA TRAZA CONTINUA")
    t = np.load(lib.NPY_ZENODO, mmap_mode="r")[0]
    tramos = [(float(t[a]), float(t[b - 1])) for a, b in lib.LIMITES_TRAMO]
    for k, (a, b) in enumerate(tramos, 1):
        p(f"   tramo {k} de la traza: {pd.to_datetime(a, unit='s')} a {pd.to_datetime(b, unit='s')} UTC")
    c = pd.read_csv(COBERTURA)
    c = c.assign(ini_unix=c.inicio_ventana_w + J2K_A_UNIX, fin_unix=c.fin_ventana_w + J2K_A_UNIX)
    tr = traslapes(c, tramos)
    p(f"   ventanas de Jose: {len(c)} ({pd.to_datetime(c.ini_unix.min(), unit='s')} a "
      f"{pd.to_datetime(c.fin_unix.max(), unit='s')} UTC); con tiempo en comun con la traza: {int(tr.sum())}")
    if tr.any():
        p(pd.crosstab(c[tr].split, c[tr].clase_principal).to_string())

    p("\n2. TEST: CAJA PRINCIPAL (AL CENTRO) Y CAJAS EXTRA (DESCENTRADAS); IoU DEL RECORTE")
    d = predicciones_test(FOLDS)
    idx = pd.read_csv(INDICE)
    if "tipo_caja" in idx.columns:
        d = d.merge(idx[["archivo", "tipo_caja", "iou"]], on="archivo", how="left")
        ev = d[d.etiqueta != "sin caja"]
        s, n = sensibilidad(ev[ev.conjunto == "recorte"], ["modelo", "tipo_caja"])
        p("   sensibilidad por clase (promedio de los 5 folds, recorte); n = recortes de test:")
        p(s.round(3).to_string())
        p(n.to_string())
        ev = ev.assign(rango_iou=pd.cut(ev.iou, [0.3, 0.5, 0.7, 1.0001], right=False,
                                        labels=["0,3-0,5", "0,5-0,7", ">=0,7"]))
        s, n = sensibilidad(ev[ev.conjunto == "recorte"], ["modelo", "rango_iou"])
        p("\n   sensibilidad por rango de IoU del recorte con su caja (recorte):")
        p(s.round(3).to_string())
        p(n.to_string())
    else:
        p("   el indice de recortes no tiene tipo_caja; se omite")

    p("\n3. TRAZA: INTERVALO DE WILSON (95 %) DE LA SENSIBILIDAD, CON EL PROMEDIO DE TEST AL LADO")
    tz = pd.read_csv(TRAZA)
    tz = tz[tz.etiqueta.isin([k for k in ec.MAPA if k != "sin caja"])]
    real = tz.etiqueta.map(ec.MAPA)
    res = pd.read_csv(os.path.join(FOLDS, "resumen_folds.csv"))
    filas = []
    for red in REDES:
        for conj in ("recorte", "fijo60s"):
            cols = [x for x in tz.columns if x.startswith(f"{red}_{conj}_fold")]
            if not cols:
                continue
            prom = res[(res.modelo == red) & (res.conjunto == conj) & (res.evaluacion == "test")
                       & (res.estadistico == "promedio")]
            for k in ec.CLASES:
                m = (real == k).to_numpy()
                ks = [int((tz.loc[m, col] == k).sum()) for col in cols]
                ic = [wilson(x, int(m.sum())) for x in ks]
                filas.append(dict(red=red, conjunto=conj, clase=k, n=int(m.sum()), traza=np.mean(ks) / m.sum(),
                                  ic_inf=np.mean([a for a, _ in ic]), ic_sup=np.mean([b for _, b in ic]),
                                  test=float(prom[f"{k}_sensibilidad"].iloc[0]) if len(prom) else np.nan))
    w = pd.DataFrame(filas)
    w["test_dentro"] = (w.test >= w.ic_inf) & (w.test <= w.ic_sup)
    p(w.round(3).to_string(index=False))
    iou_t = idx[(idx.split == "test") & idx.etiqueta.isin(["VT", "LP", "TR", "AV", "IC"])].iou
    p(f"\n   IoU de los recortes con etiqueta de clase: test mediana {iou_t.median():.2f}, >= 0,5 {np.mean(iou_t >= 0.5):.2f}; "
      f"traza mediana {tz.iou.median():.2f}, >= 0,5 {np.mean(tz.iou >= 0.5):.2f}")

    p("\n4. SIN CAJA DE TEST QUE CADA RED CLASIFICA COMO VOLCANICOS (promedio de los 5 folds)")
    sc = []
    for ruta in sorted(glob.glob(os.path.join(FOLDS, "*.json"))):
        r = json.load(open(ruta))
        if "test" not in r:
            continue
        cfg, te = r["configuracion"], r["test"]
        sc.append(dict(red=cfg["modelo"], conjunto=cfg["conjunto"], n=te["n_sin_caja"],
                       volcanicos=sum(te["sin_caja_predichos"][k] for k in ec.VOLCANICAS)))
    sc = pd.DataFrame(sc).groupby(["red", "conjunto"]).mean()
    sc["proporcion"] = sc.volcanicos / sc.n
    p(sc.round(3).to_string())

    mt = pd.read_csv(os.path.join(FOLDS, "matrices_test.csv"))
    mats = {}
    for (red, conj), g in mt.groupby(["modelo", "conjunto"]):
        mats[(red, conj)] = g.set_index("clase_real").loc[ec.CLASES, [f"pred_{k}" for k in ec.CLASES]].to_numpy()
    matrices(plt, mats, "Matrices de confusión en test, normalizadas por fila (5 folds)",
             os.path.join(SALIDA, "matrices_test.png"))
    mm = pd.read_csv(MODELOS_TRAZA)
    mats = {(red, conj): np.sum([np.array(json.loads(x)) for x in g.matriz], axis=0)
            for (red, conj), g in mm.groupby(["modelo", "conjunto"])}
    matrices(plt, mats, "Matrices de confusión en la traza continua, normalizadas por fila (5 modelos)",
             os.path.join(SALIDA, "matrices_traza.png"))
    with open(os.path.join(SALIDA, "comprobaciones.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nTexto y figuras en {SALIDA} (comprobaciones.txt, matrices_test.png, matrices_traza.png)")


if __name__ == "__main__":
    main()
