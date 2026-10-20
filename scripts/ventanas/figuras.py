#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figura del barrido de 5 minutos, usada por test_barrido.py y ventanas_traza_real.py.
"""

import numpy as np

AZUL, NARANJA, VERDE, GRIS = "#1f77b4", "#ff7f0e", "#2ca02c", "0.55"


def figura_barrido(t, x, ref, color_ref, etiqueta_ref, ventanas, disparos,
                   contenedoras, fusionados, recorte_adapt, recorte_fijo,
                   titulo, destino):
    """Dibuja la figura del barrido de 5 minutos para un evento."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(10, 3.2 + 0.28 * len(ventanas)), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 0.35 * len(ventanas) + 0.6]})

    esc = float(np.percentile(np.abs(x), 99.5)) or 1.0
    ax.plot(t, np.clip(x, -esc, esc), lw=0.3, color="0.25")
    ax.axvspan(ref[0], ref[1], color=color_ref, alpha=0.15, label=etiqueta_ref)
    y_fus, y_ad, y_fi = 1.55 * esc, 1.30 * esc, 1.12 * esc
    for i, (a, b) in enumerate(fusionados):
        ax.plot([a, b], [y_fus, y_fus], color=VERDE, lw=5, solid_capstyle="butt",
                label="detección fusionada" if i == 0 else None)
    if recorte_adapt:
        ax.plot(recorte_adapt, [y_ad, y_ad], color=AZUL, lw=5,
                solid_capstyle="butt", label="recorte adaptativo")
    if recorte_fijo:
        ax.plot(recorte_fijo, [y_fi, y_fi], color=GRIS, lw=5,
                solid_capstyle="butt", label="ventana fija de 81,92 s")
    ax.set_ylim(-1.1 * esc, 1.75 * esc)
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.9)
    ax.set_title(titulo, fontsize=8)

    for fila, (k, a, b) in enumerate(ventanas):
        y = len(ventanas) - 1 - fila
        bx.barh(y, b - a, left=a, height=0.62, color="0.93",
                edgecolor="0.3" if k in contenedoras else "0.7",
                lw=1.4 if k in contenedoras else 0.6)
        for s, e, cortado in disparos.get(k, []):
            bx.barh(y, e - s, left=s, height=0.62,
                    color=NARANJA if cortado else AZUL)
    bx.axvspan(ref[0], ref[1], color=color_ref, alpha=0.10)
    bx.set_yticks(range(len(ventanas)))
    bx.set_yticklabels([f"V{k}" + (" *" if k in contenedoras else "")
                        for k, _, _ in reversed(ventanas)], fontsize=6)
    bx.set_xlabel("tiempo relativo al inicio del evento de referencia (s)")
    bx.set_ylabel("ventanas de 300 s", fontsize=7)
    bx.text(1.0, -0.05 - 1.2 / (len(ventanas) + 2),
            "* contiene el evento entero.  Disparos: azul, entero en la ventana;"
            " naranja, cortado en un borde", transform=bx.transAxes,
            ha="right", va="top", fontsize=6)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
