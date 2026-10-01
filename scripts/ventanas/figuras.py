#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figuras.py
==========

Figura del barrido de 5 minutos, compartida por test_barrido.py (casos
sinteticos) y ventanas_traza_real.py (traza real).

Arriba: la onda, el intervalo de referencia, la deteccion fusionada, el
recorte adaptativo y la ventana fija de 81,92 s.
Abajo: una barra por cada ventana de 300 s que toca el evento, con los
disparos del detector dentro de esa ventana. Azul si el disparo cae entero
dentro de la ventana; naranja si esta cortado en un borde.

Todos los tiempos se dibujan en segundos relativos al inicio del evento de
referencia.
"""

import numpy as np

AZUL, NARANJA, VERDE, GRIS = "#1f77b4", "#ff7f0e", "#2ca02c", "0.55"


def figura_barrido(t, x, ref, color_ref, etiqueta_ref, ventanas, disparos,
                   contenedoras, fusionados, recorte_adapt, recorte_fijo,
                   titulo, destino):
    """
    t, x          : tiempo relativo (s) y senal del tramo visible.
    ref           : (ini_s, fin_s) del evento de referencia, relativo.
    ventanas      : lista de (k, ini_s, fin_s) de las ventanas que lo tocan.
    disparos      : dict k -> lista de (ini_s, fin_s, cortado).
    contenedoras  : conjunto de k que contienen el evento entero.
    fusionados    : lista de (ini_s, fin_s) tras la fusion.
    recorte_adapt, recorte_fijo : (ini_s, fin_s) o None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(10, 3.2 + 0.28 * len(ventanas)), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 0.35 * len(ventanas) + 0.6]})

    # --- panel superior: onda y resultado ------------------------------
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

    # --- panel inferior: una barra por ventana -------------------------
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
