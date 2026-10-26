#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curvas por umbral con la deteccion por interseccion: sensibilidad por clase y FDR (fig8) y
sensibilidad volcanica contra falsas alarmas por hora (fig9), cada red con las ventanas de
su configuracion. Lee las tablas de grilla_fre3c.py y configuracion_fre3c.py.

Uso: python figuras_interseccion_fre3c.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fre3c_lib as lib

# paleta categorica validada (orden fijo); el total volcanico y la FDR van en tinta
COLOR = {"VT": "#2a78d6", "LP": "#eb6834", "TR": "#1baf7a"}
MARCA = {"VT": "o", "LP": "s", "TR": "^"}
COLOR_DET = {"PhaseNet P y S": "#2a78d6", "EQTransformer deteccion": "#eb6834",
             "EQTransformer P y S": "#1baf7a", "STA/LTA": "#0b0b0b"}
NOMBRE = {"PhaseNet P y S": "PhaseNet, P y S", "EQTransformer deteccion": "EQTransformer, detección",
          "EQTransformer P y S": "EQTransformer, P y S", "STA/LTA": "STA/LTA"}
TINTA, TINTA2, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# etiqueta de cada configuracion elegida: arriba PhaseNet, abajo las demas, para que no se tapen
DESPLAZA = {"PhaseNet P y S": (6, 6), "EQTransformer deteccion": (6, -14),
            "EQTransformer P y S": (6, -14), "STA/LTA": (6, -14)}
CORTO = {"PhaseNet P y S": "PhaseNet", "EQTransformer deteccion": "EQTransformer detección",
         "EQTransformer P y S": "EQTransformer P y S", "STA/LTA": "STA/LTA"}


def coma(x, _=None):
    return f"{x:g}".replace(".", ",")


def estilo(ax):
    from matplotlib.ticker import FuncFormatter
    ax.grid(axis="y", color=REJILLA, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TINTA2)
    ax.tick_params(colors=TINTA2, labelsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(coma))
    ax.yaxis.set_major_formatter(FuncFormatter(coma))


def filas_config(g, flujo, cfg):
    m = (g.detector == flujo) & (g.overlap == cfg["overlap"]) & (g.blinding == cfg["blinding"]) \
        & (g.stacking == cfg["stacking"])
    return g[m]


def panel(ax, d, x, titulo, x_elegido, etiqueta_x):
    d = d.sort_values(x)
    for c in lib.VOLCANICAS:
        ax.plot(d[x], d[f"k_{c}"] / d[f"n_{c}"], color=COLOR[c], lw=1.6, marker=MARCA[c], ms=3.5,
                label=c)
    ax.plot(d[x], d.sens_volc, color=TINTA, lw=2.4, label="Volcánicos")
    ax.plot(d[x], d.fdr, color=TINTA2, lw=1.6, ls="--", label="FDR")
    ax.axhline(0.95, color=TINTA2, lw=0.8, ls=":")
    ax.axvline(x_elegido, color=TINTA2, lw=0.8, ls=":")
    ax.text(x_elegido + 0.01, 0.03, f"elegido: {coma(x_elegido)}", color=TINTA2, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(titulo, color=TINTA, fontsize=9, loc="left")
    ax.set_xlabel(etiqueta_x, color=TINTA2, fontsize=8)
    estilo(ax)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = pd.read_csv(os.path.join(lib.DIR_SALIDA, "grilla_barrido.csv"))
    g = g[g.tramo.astype(str) == "todos"]
    with open(os.path.join(lib.DIR_SALIDA, "configuracion_elegida.json")) as f:
        cfgs = json.load(f)
    clases = pd.read_csv(os.path.join(lib.DIR_SALIDA, "config_clases.csv"))

    pn, eqd, eqps = (filas_config(g, f, cfgs[f]) for f in
                     ("PhaseNet P y S", "EQTransformer deteccion", "EQTransformer P y S"))
    cp, cd, ce = cfgs["PhaseNet P y S"], cfgs["EQTransformer deteccion"], cfgs["EQTransformer P y S"]

    # fig8: en cada panel se mueve un umbral y el otro queda en el valor elegido
    fig, ejes = plt.subplots(2, 3, figsize=(10.5, 6.6), sharey=True)
    paneles = [
        (pn[np.isclose(pn.S_threshold, cp["S_threshold"])], "P_threshold",
         f"PhaseNet: P_threshold (S en {coma(cp['S_threshold'])})", cp["P_threshold"]),
        (pn[np.isclose(pn.P_threshold, cp["P_threshold"])], "S_threshold",
         f"PhaseNet: S_threshold (P en {coma(cp['P_threshold'])})", cp["S_threshold"]),
        (eqd, "detection_threshold", "EQTransformer: detection_threshold", cd["detection_threshold"]),
        (eqps[np.isclose(eqps.S_threshold, ce["S_threshold"])], "P_threshold",
         f"EQTransformer: P_threshold (S en {coma(ce['S_threshold'])})", ce["P_threshold"]),
        (eqps[np.isclose(eqps.P_threshold, ce["P_threshold"])], "S_threshold",
         f"EQTransformer: S_threshold (P en {coma(ce['P_threshold'])})", ce["S_threshold"]),
    ]
    for ax, (d, x, titulo, xe) in zip(ejes.flat, paneles):
        panel(ax, d, x, titulo, xe, "umbral")
    for ax in ejes[:, 0]:
        ax.set_ylabel("sensibilidad y FDR", color=TINTA2, fontsize=8)
    leyenda = ejes.flat[5]
    leyenda.axis("off")
    manejadores, nombres = ejes.flat[0].get_legend_handles_labels()
    leyenda.legend(manejadores, nombres, frameon=False, fontsize=9, loc="center")
    fig.tight_layout()
    ruta8 = os.path.join(lib.DIR_SALIDA, "fig8_curvas_interseccion.png")
    fig.savefig(ruta8, dpi=150, facecolor=FONDO)
    plt.close(fig)

    # fig9: cada punto es una combinacion de umbrales con las ventanas de cada configuracion
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axhline(0.95, color=TINTA2, lw=0.8, ls=":")
    ax.text(1, 0.955, "meta 0,95", color=TINTA2, fontsize=7)
    for flujo, d in (("PhaseNet P y S", pn), ("EQTransformer P y S", eqps), ("EQTransformer deteccion", eqd)):
        ax.scatter(d.fp_h, d.sens_volc, s=10, color=COLOR_DET[flujo], alpha=0.35, linewidths=0)
        cfg = cfgs[flujo]
        e = d
        for k in ("P_threshold", "S_threshold", "detection_threshold"):
            e = e[e[k].isna()] if cfg[k] is None else e[np.isclose(e[k], cfg[k])]
        if len(e):
            r = e.iloc[0]
            ax.scatter([r.fp_h], [r.sens_volc], s=70, color=COLOR_DET[flujo], edgecolors=FONDO,
                       linewidths=1.5, zorder=3)
            ax.annotate(f"{CORTO[flujo]}: {coma(round(r.sens_volc, 3))}; {coma(round(r.fp_h, 1))} FP/h",
                        (r.fp_h, r.sens_volc), xytext=DESPLAZA[flujo], textcoords="offset points",
                        fontsize=7, color=TINTA)
    s = clases[clases.detector == "STA/LTA"].iloc[0]
    sens_s = s.k_volcanico / s.n_volcanico
    ax.scatter([s.fp_h], [sens_s], s=70, marker="D", color=COLOR_DET["STA/LTA"], edgecolors=FONDO,
               linewidths=1.5, zorder=3)
    ax.annotate(f"STA/LTA: {coma(round(sens_s, 3))}; {coma(round(s.fp_h, 1))} FP/h", (s.fp_h, sens_s),
                xytext=DESPLAZA["STA/LTA"], textcoords="offset points", fontsize=7, color=TINTA)
    ax.set_xlabel("falsas alarmas por hora", color=TINTA2, fontsize=9)
    ax.set_ylabel("sensibilidad volcánica", color=TINTA2, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    estilo(ax)
    from matplotlib.lines import Line2D
    marcas = [Line2D([], [], ls="", marker="o", ms=6, color=COLOR_DET[f], label=NOMBRE[f])
              for f in ("PhaseNet P y S", "EQTransformer P y S", "EQTransformer deteccion")]
    marcas.append(Line2D([], [], ls="", marker="D", ms=6, color=COLOR_DET["STA/LTA"], label="STA/LTA (Etapa 2)"))
    ax.legend(handles=marcas, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    ruta9 = os.path.join(lib.DIR_SALIDA, "fig9_costo_interseccion.png")
    fig.savefig(ruta9, dpi=150, facecolor=FONDO)
    plt.close(fig)
    print(f"Figuras: {ruta8} y {ruta9}")


if __name__ == "__main__":
    main()
