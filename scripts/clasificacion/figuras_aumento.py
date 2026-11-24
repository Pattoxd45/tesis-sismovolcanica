#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figuras para acordar con el profesor el tamano de entrada y el aumento de datos, con recortes reales:
1) estiramiento paso a paso: un recorte corto (menos columnas que el lado de entrada) y uno largo (mas),
   desde la ventana de 300 s hasta la imagen de 224 x 224 que recibe la red;
2) un evento con distintos rangos de dB (maximo corrido y ancho del rango), el "color jitter" que pidio;
3) el mismo evento con ElasticTransform de torchvision, en los dos ejes y solo en el eje del tiempo.

Uso: python figuras_aumento.py [--corto <archivo>] [--largo <archivo>] [--evento <archivo>]
     [--desplazamientos -10 0 10] [--anchos 45 60 75] [--alfa 50] [--sigma 5]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import butter
from torchvision.transforms import v2

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "sistema"))
import entrenar_cnn as ec
import espectrogramas_recortes as er

LADO = ec.LADO
INDICE = os.path.join(er.rv.DIR_SALIDA, "espectrogramas_recortes.csv")
SALIDA = os.path.join(er.lib.DIR_TESIS, "resultados", "clasificacion")
COMPONENTES = ("Z", "N", "E")


def coma(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def elegir(idx):
    """Recortes de desarrollo con etiqueta de Jose: un VT de duracion mediana (corto), el recorte con un numero de
    columnas mas cercano a 1,5 veces el lado (largo) y un LP de duracion mediana (evento para el aumento)."""
    d = idx[(idx.split == "desarrollo") & idx.etiqueta.isin(["VT", "LP", "TR"])].sort_values("archivo")

    def mediano(et, cond=True):
        x = d[(d.etiqueta == et) & cond]
        return None if x.empty else x.loc[(x.dur_s - x.dur_s.median()).abs().idxmin()]

    largos = d[d.columnas > LADO]
    largo = None if largos.empty else largos.loc[(largos.columnas - 1.5 * LADO).abs().idxmin()]
    return dict(corto=mediano("VT", d.columnas < LADO), largo=largo, evento=mediano("LP"))


def ventana(v, dir_ventanas, sos):
    """Espectrograma en dB de Z, N y E de la ventana y el rango de cada uno (como en las imagenes de entrenamiento)."""
    w = np.load(os.path.join(dir_ventanas, f"{v}_zne.npz"))
    esp = {c: er.espectrograma(w[c], sos) for c in COMPONENTES}
    return {c: x[0] for c, x in esp.items()}, {c: (x[1], x[2]) for c, x in esp.items()}


def con_rango(lims, desplazamiento, ancho):
    """Rango de cada componente con el maximo corrido y otro ancho: el azul mas profundo es el minimo y el rojo el maximo."""
    return {c: (vmax + desplazamiento - ancho, vmax + desplazamiento) for c, (_, vmax) in lims.items()}


def a_lado(arr):
    """Igual que Recortes de entrenar_cnn.py con 'estirar': PIL BILINEAR a LADO x LADO."""
    return np.asarray(Image.fromarray(arr).resize((LADO, LADO), Image.BILINEAR))


def elastica(arr, alfa, sigma, semilla):
    """ElasticTransform de torchvision sobre la imagen de LADO x LADO; alfa es (tiempo, frecuencia)."""
    torch.manual_seed(semilla)
    x = torch.from_numpy(arr.copy()).permute(2, 0, 1)
    return v2.ElasticTransform(alpha=alfa, sigma=sigma)(x).permute(1, 2, 0).numpy()


def ejes_componentes(a, alto):
    a.set_yticks([alto * (i + 0.5) / 3 for i in range(3)], list(COMPONENTES))
    a.tick_params(length=0, labelsize=8)


def figura_estiramiento(plt, casos, ruta):
    """Por fila: ventana con el recorte marcado, recorte a su tamano y entrada de la red, a la misma escala de pixeles."""
    n_col = next(iter(casos.values()))[1]["Z"].shape[1]
    fig = plt.figure(figsize=(15, 3.4 * len(casos)))
    sub = fig.subfigures(len(casos), 1)
    for s, (titulo, (r, dbs, lims, col)) in zip(np.atleast_1d(sub), casos.items()):
        rec = er.imagen(dbs, lims, col)
        ax = s.subplots(1, 3, gridspec_kw=dict(width_ratios=[n_col, max(rec.shape[1], 40), LADO]))
        ax[0].imshow(er.imagen(dbs, lims, np.arange(n_col)), interpolation="nearest")
        for c in (col[0] - 0.5, col[-1] + 0.5):
            ax[0].axvline(c, color="white", lw=1.5)
        ax[0].set_title(f"1. Ventana de 300 s: {n_col} columnas; el recorte, entre las líneas blancas", fontsize=9)
        ax[1].imshow(rec, interpolation="nearest")
        ax[1].set_title(f"2. Recorte: {rec.shape[1]} x {rec.shape[0]}", fontsize=9)
        ax[2].imshow(a_lado(rec), interpolation="nearest")
        verbo = "estira" if rec.shape[1] < LADO else "comprime"
        ax[2].set_title(f"3. Entrada de la red: {LADO} x {LADO}\n(el ancho se {verbo} x{coma(LADO / rec.shape[1], 2)})", fontsize=9)
        for a, alto in zip(ax, (rec.shape[0], rec.shape[0], LADO)):
            ejes_componentes(a, alto)
            a.set_xticks([])
        s.suptitle(f"{titulo}: {r.etiqueta}, {coma(r.dur_s)} s ({r.archivo})", fontsize=10, x=0.01, ha="left")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_rangos(plt, r, dbs, lims, col, desplazamientos, anchos, ruta):
    fig, ax = plt.subplots(len(desplazamientos), len(anchos), figsize=(2.6 * len(anchos), 2.8 * len(desplazamientos)),
                           squeeze=False)
    for i, d in enumerate(desplazamientos):
        for j, w in enumerate(anchos):
            a = ax[i, j]
            a.imshow(a_lado(er.imagen(dbs, con_rango(lims, d, w), col)), interpolation="nearest")
            jose = d == 0 and w == er.RANGO_DB
            a.set_title(f"máximo {f'{d:+g}' if d else '0'} dB, ancho {w:g} dB" + ("\n(rango de José)" if jose else ""), fontsize=8,
                        fontweight="bold" if jose else "normal")
            ejes_componentes(a, LADO)
            a.set_xticks([])
    fig.suptitle(f"{r.etiqueta} de {coma(r.dur_s)} s ({r.archivo}) con distintos rangos de dB. Cada componente usa el "
                 f"máximo de su ventana;\nel azul más profundo es (máximo - ancho) y el rojo es el máximo", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_elastica(plt, r, dbs, lims, col, alfa, sigma, ruta, n=3):
    base = a_lado(er.imagen(dbs, lims, col))
    filas = [(f"dos ejes: alpha {alfa:g}, sigma {sigma:g} (por defecto)", (alfa, alfa)),
             (f"solo tiempo: alpha ({alfa:g}, 0), sigma {sigma:g}", (alfa, 0.0))]
    fig, ax = plt.subplots(len(filas), n + 1, figsize=(2.6 * (n + 1), 2.9 * len(filas)), squeeze=False)
    for i, (titulo, a_xy) in enumerate(filas):
        ax[i, 0].imshow(base, interpolation="nearest")
        ax[i, 0].set_title("original", fontsize=8)
        for k in range(n):
            ax[i, k + 1].imshow(elastica(base, a_xy, sigma, semilla=k), interpolation="nearest")
            ax[i, k + 1].set_title(f"{titulo}\nsorteo {k + 1}", fontsize=7)
        for a in ax[i]:
            ejes_componentes(a, LADO)
            a.set_xticks([])
    fig.suptitle(f"ElasticTransform de torchvision sobre {r.etiqueta} de {coma(r.dur_s)} s ({r.archivo}); "
                 f"lo que sale de la imagen queda en negro (fill 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corto", help="archivo del recorte corto (por defecto, un VT de duracion mediana)")
    ap.add_argument("--largo", help="archivo del recorte largo (por defecto, el mas cercano a 1,5 x 224 columnas)")
    ap.add_argument("--evento", help="archivo del evento para el aumento (por defecto, un LP de duracion mediana)")
    ap.add_argument("--desplazamientos", type=float, nargs="+", default=[-10.0, 0.0, 10.0],
                    help="corrimiento del maximo del rango, en dB")
    ap.add_argument("--anchos", type=float, nargs="+", default=[45.0, er.RANGO_DB, 75.0], help="ancho del rango, en dB")
    ap.add_argument("--alfa", type=float, default=50.0, help="alpha de ElasticTransform (por defecto de torchvision)")
    ap.add_argument("--sigma", type=float, default=5.0, help="sigma de ElasticTransform (por defecto de torchvision)")
    ap.add_argument("--indice", default=INDICE)
    ap.add_argument("--ventanas", default=er.ej.DIR_SALIDA, help="carpeta con <ventana>_zne.npz")
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = pd.read_csv(args.indice)
    elegidos = elegir(idx)
    for clave in ("corto", "largo", "evento"):
        a = getattr(args, clave)
        if a:
            elegidos[clave] = idx.set_index("archivo", drop=False).loc[a]
    sos = butter(er.ORDEN_FILTRO, list(er.BANDA_FILTRO), btype="bandpass", fs=er.FS, output="sos")
    datos = {}
    for clave, r in elegidos.items():
        if r is None:
            print(f"  sin recorte {clave}: no hay uno que cumpla la condicion")
            continue
        dbs, lims = ventana(r.ventana, args.ventanas, sos)
        datos[clave] = (r, dbs, lims, er.columnas(r.ini_s * er.FS, r.fin_s * er.FS, dbs["Z"].shape[1]))
        print(f"  {clave}: {r.archivo} ({r.etiqueta}, {r.dur_s:.1f} s, {datos[clave][3].size} columnas, split {r.split})")

    os.makedirs(args.salida, exist_ok=True)
    rutas = []
    casos = {t: datos[k] for k, t in (("corto", "Recorte corto"), ("largo", "Recorte largo")) if k in datos}
    if casos:
        rutas.append(os.path.join(args.salida, "figura_estiramiento.png"))
        figura_estiramiento(plt, casos, rutas[-1])
    if "evento" in datos:
        rutas.append(os.path.join(args.salida, "figura_rangos_db.png"))
        figura_rangos(plt, *datos["evento"], args.desplazamientos, args.anchos, rutas[-1])
        rutas.append(os.path.join(args.salida, "figura_elastica.png"))
        figura_elastica(plt, *datos["evento"], args.alfa, args.sigma, rutas[-1])
    print("Figuras en:\n  " + "\n  ".join(rutas))


if __name__ == "__main__":
    main()
