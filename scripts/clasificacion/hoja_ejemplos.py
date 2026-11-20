#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hojas para revisar a ojo las predicciones de un fold: recortes sin caja que las tres redes llaman
volcanicos, sin caja que las tres rechazan y confusiones de EfficientNet-B0 entre LP, TR y VT. Cada
fila muestra el espectrograma de Z de la ventana completa con las cajas de Jose y el recorte marcado,
y a la derecha la imagen que vio la red (Z, N y E). Los ejemplos de cada grupo se toman con una
semilla fija, no a mano.

Uso: python hoja_ejemplos.py --predicciones <carpeta con los *_predicciones.csv> [--n 8] [--semilla 0]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
REDES = ["efficientnet_b0", "resnet50", "vgg16"]
CORTO = {"efficientnet_b0": "EffNet", "resnet50": "ResNet", "vgg16": "VGG16"}
VOLCANICAS = ["VT", "LP", "TR"]
CLASES = VOLCANICAS + ["rechazo"]


def unir(carpeta, fold=1):
    """Una fila por recorte con la prediccion de cada red y la probabilidad de la clase que eligio."""
    d = None
    for m in REDES:
        rutas = glob.glob(os.path.join(carpeta, f"{m}_fold{fold}_*_sin_falsas_predicciones.csv"))
        if len(rutas) != 1:
            raise SystemExit(f"DETENIDO: se esperaba un archivo de predicciones de {m} en {carpeta}; hay {len(rutas)}")
        p = pd.read_csv(rutas[0])
        q = p[["archivo", "ventana", "etiqueta", "clase"]].copy()
        q[f"pred_{m}"] = p.prediccion
        q[f"prob_{m}"] = p[[f"p_{c}" for c in CLASES]].max(axis=1)
        if d is None:
            d = q
        else:
            if not (d.archivo.to_numpy() == q.archivo.to_numpy()).all():
                raise SystemExit(f"DETENIDO: los recortes de {m} no son los mismos que los de {REDES[0]}")
            d = d.join(q[[f"pred_{m}", f"prob_{m}"]])
    return d


def grupos(d):
    """Grupos que se revisan; las confusiones son las de EfficientNet-B0."""
    sc = d.etiqueta == "sin caja"
    n_volc = sum(d[f"pred_{m}"].isin(VOLCANICAS).astype(int) for m in REDES)
    e = d.pred_efficientnet_b0
    return {
        "Sin caja: las tres redes dicen volcanico": sc & (n_volc == 3),
        "Sin caja: las tres redes dicen rechazo": sc & (n_volc == 0),
        "LP real; EfficientNet-B0 dice TR": (d.clase == "LP") & (e == "TR"),
        "TR real; EfficientNet-B0 dice LP": (d.clase == "TR") & (e == "LP"),
        "LP real; EfficientNet-B0 dice VT": (d.clase == "LP") & (e == "VT"),
    }


def fila(axs, r, rec, cajas, er, sos):
    import matplotlib.pyplot as plt

    w = np.load(os.path.join(er.ej.DIR_SALIDA, f"{r.ventana}_zne.npz"))
    db, vmin, vmax = er.espectrograma(w["Z"], sos)
    dur = w["Z"].size / er.FS
    a = axs[0]
    a.imshow(db, origin="lower", aspect="auto", cmap=er.CMAP, vmin=vmin, vmax=vmax, extent=[0, dur, *er.BANDA_IMAGEN])
    for c in cajas.itertuples():
        x0, x1 = c.ini / er.FS, c.fin / er.FS
        a.add_patch(plt.Rectangle((x0, 0.3), x1 - x0, 14.4, fill=False, ec="white", lw=1.2))
        a.text(x0 + 1, 13.6, c.clase, color="white", fontsize=8, fontweight="bold", va="top")
    a.axvspan(rec.ini_s, rec.fin_s, color="magenta", alpha=0.25, lw=0)
    a.axvline(rec.ini_s, color="magenta", lw=1.2)
    a.axvline(rec.fin_s, color="magenta", lw=1.2)
    a.set_xlim(0, dur)
    a.set_ylabel("Hz", fontsize=8)
    a.tick_params(labelsize=7)
    a.set_title(f"{r.ventana}, recorte {int(rec.k)} ({rec.dur_s:.1f} s); etiqueta: {r.etiqueta}", fontsize=8, loc="left")
    b = axs[1]
    b.imshow(plt.imread(os.path.join(er.DIR_IMAGENES, r.archivo)), aspect="auto")
    b.axis("off")
    b.set_title("\n".join(f"{CORTO[m]}: {r[f'pred_{m}']} {r[f'prob_{m}']:.2f}" for m in REDES), fontsize=7.5)


def hoja(sel, idx, cajas, er, sos, titulo, ruta):
    import matplotlib.pyplot as plt

    # una fila angosta con el nombre del grupo antes de sus ejemplos
    alto = [h for _, x in sel for h in [0.3] + [1.0] * len(x)]
    fig, ax = plt.subplots(len(alto), 2, figsize=(14, 2.4 * sum(alto)),
                           gridspec_kw=dict(width_ratios=[4, 1], height_ratios=alto))
    i = 0
    for nombre, x in sel:
        for a in ax[i]:
            a.axis("off")
        ax[i, 0].text(0, 0.2, nombre, fontsize=11, fontweight="bold", transform=ax[i, 0].transAxes)
        i += 1
        for _, r in x.iterrows():
            fila(ax[i], r, idx.loc[r.archivo], cajas.get(r.ventana, pd.DataFrame()), er, sos)
            i += 1
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicciones", required=True, help="carpeta con los tres *_predicciones.csv")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--n", type=int, default=8, help="ejemplos de sin caja tomados por volcanicos")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from scipy.signal import butter
    sys.path.insert(0, os.path.join(AQUI, "..", "sistema"))
    import espectrogramas_recortes as er

    d = unir(os.path.expanduser(args.predicciones), args.fold)
    g = grupos(d)
    print(f"{len(d)} recortes del fold {args.fold}: {int((d.etiqueta != 'sin caja').sum())} eventos de Jose y "
          f"{int((d.etiqueta == 'sin caja').sum())} sin caja")
    for nombre, mask in g.items():
        print(f"  {nombre}: {int(mask.sum())}")
    n_volc = sum(d[f"pred_{m}"].isin(VOLCANICAS).astype(int) for m in REDES)[d.etiqueta == "sin caja"]
    print("  Sin caja segun cuantas redes dicen volcanico (0 a 3):", n_volc.value_counts().sort_index().to_dict())

    idx = pd.read_csv(os.path.join(er.rv.DIR_SALIDA, "espectrogramas_recortes.csv")).set_index("archivo")
    cajas = er.rv.cargar_cajas()
    sos = butter(er.ORDEN_FILTRO, list(er.BANDA_FILTRO), btype="bandpass", fs=er.FS, output="sos")

    def tomar(nombre, k):
        return d[g[nombre]].sample(min(k, int(g[nombre].sum())), random_state=args.semilla)

    nombres = list(g)
    salida = os.path.join(er.lib.DIR_TESIS, "resultados", "clasificacion")
    os.makedirs(salida, exist_ok=True)
    r1 = os.path.join(salida, f"hoja_sin_caja_fold{args.fold}.png")
    hoja([(nombres[0], tomar(nombres[0], args.n)), (nombres[1], tomar(nombres[1], 2))], idx, cajas, er, sos,
         "Recortes sin caja (Z de la ventana, eje en segundos: blanco, cajas de Jose; magenta, el recorte)", r1)
    r2 = os.path.join(salida, f"hoja_confusiones_fold{args.fold}.png")
    hoja([(x, tomar(x, 2)) for x in nombres[2:]], idx, cajas, er, sos,
         "Confusiones de EfficientNet-B0 (Z de la ventana, eje en segundos: blanco, cajas de Jose; magenta, el recorte)", r2)
    print(f"\nHojas en {r1}\n         {r2}")


if __name__ == "__main__":
    main()
