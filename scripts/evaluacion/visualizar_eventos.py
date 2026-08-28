#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizar_eventos.py
=====================

Dos cosas distintas, ambas necesarias:

(A) FIGURA DE VALIDACION -- forma de onda con el intervalo del catalogo y
    lo que detecto cada detector encima. Sirve para ver con los ojos si
    los numeros de la evaluacion significan lo que uno cree, y para las
    figuras del capitulo de resultados.

(B) REPRESENTACION DE ENTRADA DEL CLASIFICADOR -- espectrograma + forma
    de onda, que es lo que realmente consume el VGG16 (representacion
    tipo 2 del codigo de Curilem). El clasificador no ve la senal: ve
    una imagen.

    IMPORTANTE: los parametros del espectrograma aqui son razonables pero
    NO estan verificados contra el codigo de Curilem. Antes de generar el
    conjunto de entrenamiento hay que copiarlos exactamente de su
    repositorio, o el clasificador se entrenaria sobre imagenes distintas
    de las que uso el trabajo de referencia.

Uso:
    python visualizar_eventos.py                      # 1 evento por clase
    python visualizar_eventos.py --clases TR IC
    python visualizar_eventos.py --n 3 --canal 2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_detectores import (FS, CORTES, NPY, CSV, SALIDA, CLASES,
                                STALTA, segmentos, fusionar_canal)

FIGURAS = os.path.join(SALIDA, "figuras")
CACHE_PICKS = os.path.join(SALIDA, "cache_picks")

COLOR = {"VT": "#df8d5e", "LP": "#2ca02c", "TR": "#d62728",
         "AV": "#9467bd", "IC": "#8c564b"}

# Espectrograma: valores iniciales, PENDIENTE de verificar contra Curilem
NPERSEG = 256
NOVERLAP = 128


def stalta_ventana(x, cfg=STALTA, descartar=0):
    """
    STA/LTA sobre una ventana. Devuelve intervalos relativos a la muestra
    `descartar` (el contexto previo se usa solo para calentar la LTA).

    OJO: la LTA de 60 s necesita al menos ese tiempo de senal antes de dar
    una razon valida. Calcular STA/LTA sobre una ventana corta centrada en
    el evento produce CERO disparos, aunque el detector si funcione sobre
    la traza completa. Por eso hay que alimentarlo con contexto previo.
    """
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from obspy.signal.filter import bandpass

    if np.allclose(x, 0) or len(x) < int(cfg["lta_s"] * FS) * 2:
        return []
    xf = bandpass(x, cfg["banda"][0], cfg["banda"][1], df=FS,
                  corners=4, zerophase=True)
    cft = classic_sta_lta(xf, int(cfg["sta_s"] * FS), int(cfg["lta_s"] * FS))
    cft = np.nan_to_num(cft, nan=0.0, posinf=0.0, neginf=0.0)
    trg = [(int(s), int(e)) for s, e in
           trigger_onset(cft, cfg["thr_on"], cfg["thr_off"])]
    trg = fusionar_canal(trg, int(5.0 * FS))
    # trasladar al origen de la ventana visible y recortar
    out = []
    for s, e in trg:
        s2, e2 = s - descartar, e - descartar
        if e2 > 0:
            out.append((max(0, s2), e2))
    return out


def cargar_picks(nombre, modo, canal, ini, fin, conf_min=0.2):
    """Lee picks cacheados por barrido_modelos.py dentro de una ventana."""
    ruta = os.path.join(CACHE_PICKS, f"picks_{nombre}_{modo}.csv")
    if not os.path.exists(ruta):
        return []
    df = pd.read_csv(ruta)
    sel = df[(df.canal == canal) & (df.muestra >= ini) &
             (df.muestra < fin) & (df.conf >= conf_min)]
    return [(int(r.muestra) - ini, str(r.fase), float(r.conf))
            for r in sel.itertuples()]


def figura_validacion(a, ev, canal, destino, catalogo=None):
    """Panel A: onda + catalogo + detecciones de cada detector."""
    s, e = int(ev.idx_start), int(ev.idx_end)
    dur = e - s
    pad = max(int(30 * FS), dur // 2)
    ini, fin = max(0, s - pad), min(a.shape[-1], e + pad)
    x = np.asarray(a[canal, ini:fin], dtype=np.float64)
    t = np.arange(len(x)) / FS

    # contexto previo para que la LTA de 60 s este caliente al llegar
    # a la ventana visible; si no, STA/LTA no dispara nunca
    lead = int(3 * STALTA["lta_s"] * FS)
    ini_ext = max(0, ini - lead)
    x_ext = np.asarray(a[canal, ini_ext:fin], dtype=np.float64)
    trigs = stalta_ventana(x_ext, descartar=ini - ini_ext)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(t, x, color="black", lw=0.6)
    ax.axvspan((s - ini) / FS, (e - ini) / FS, color=COLOR.get(ev.event_type,
                                                               "gray"),
               alpha=0.25, label=f"catalogo {ev.event_type} ({dur / FS:.1f} s)")

    # otros eventos del catalogo dentro de la ventana: si aparece energia
    # sin sombrear, el catalogo no es exhaustivo y eso cambia la lectura
    # de la precision de todos los detectores
    otros = 0
    if catalogo is not None:
        vecinos = catalogo[(catalogo.idx_end > ini) &
                           (catalogo.idx_start < fin) &
                           (catalogo.idx_start != s)]
        for r in vecinos.itertuples():
            a0 = (max(int(r.idx_start), ini) - ini) / FS
            a1 = (min(int(r.idx_end), fin) - ini) / FS
            ax.axvspan(a0, a1, color=COLOR.get(r.event_type, "gray"),
                       alpha=0.15, hatch="//", ec="gray")
            ax.text((a0 + a1) / 2, ax.get_ylim()[1] * 0.85, r.event_type,
                    fontsize=7, ha="center", color="gray")
            otros += 1

    ax.set_ylabel("amplitud [cuentas]")
    ax.set_title(f"{ev.event_type} | muestras {s}-{e} | "
                 f"duracion {dur / FS:.1f} s | canal {canal} | "
                 f"otros eventos catalogados en la ventana: {otros}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, ls="--")

    ax = axes[1]
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["STA/LTA", "PhaseNet", "EQTransformer"], fontsize=9)

    if trigs:
        for s_rel, e_rel in trigs:
            ax.add_patch(Rectangle((s_rel / FS, 0.2), (e_rel - s_rel) / FS,
                                   0.6, color="#444444", alpha=0.7))
    else:
        ax.text(0.5, 0.5, "sin disparos en la ventana",
                fontsize=7, color="gray", style="italic")

    for fila, (nombre, modo, color) in enumerate(
            [("phasenet", "duplicar", "#1f77b4"),
             ("eqtransformer", "ceros", "#d62728")], start=1):
        picks = cargar_picks(nombre, modo, canal, ini, fin)
        for m_rel, fase, conf in picks:
            ax.vlines(m_rel / FS, fila + 0.15, fila + 0.85,
                      color=color, lw=1.8, alpha=min(1.0, 0.35 + conf))
            ax.text(m_rel / FS, fila + 0.88, fase, fontsize=6,
                    ha="center", color=color)
        if not picks:
            ax.text(0.5, fila + 0.5, "sin picks en la ventana",
                    fontsize=7, color="gray", style="italic")

    ax.axvspan((s - ini) / FS, (e - ini) / FS,
               color=COLOR.get(ev.event_type, "gray"), alpha=0.12)
    ax.set_xlabel("tiempo [s] desde el inicio de la ventana")
    ax.grid(alpha=0.3, ls="--", axis="x")

    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def figura_representacion(a, ev, canal, destino):
    """Panel B: lo que realmente ve el VGG16 (espectrograma + onda)."""
    from scipy.signal import spectrogram

    s, e = int(ev.idx_start), int(ev.idx_end)
    x = np.asarray(a[canal, s:e], dtype=np.float64)
    if x.size < NPERSEG or np.allclose(x, 0):
        return False

    f, tt, Sxx = spectrogram(x, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].pcolormesh(tt, f, Sxx_db, shading="gouraud", cmap="viridis")
    axes[0].set_ylabel("frecuencia [Hz]")
    axes[0].set_title(f"Entrada del clasificador | {ev.event_type} | "
                      f"{(e - s) / FS:.1f} s")
    axes[1].plot(np.arange(len(x)) / FS, x, color="black", lw=0.6)
    axes[1].set_xlabel("tiempo [s]")
    axes[1].set_ylabel("amplitud")
    axes[1].grid(alpha=0.3, ls="--")

    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clases", nargs="+", default=CLASES)
    ap.add_argument("--n", type=int, default=1, help="eventos por clase")
    ap.add_argument("--canal", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(FIGURAS, exist_ok=True)
    a = np.load(NPY, mmap_mode="r")
    df = pd.read_csv(CSV)
    df = df.assign(dur=(df.idx_end - df.idx_start) / FS)

    if not os.path.exists(CACHE_PICKS):
        print(f"AVISO: no hay picks cacheados en {CACHE_PICKS}.")
        print("       Corre barrido_modelos.py primero para ver PhaseNet "
              "y EQTransformer en las figuras.\n")

    generadas = []
    for clase in args.clases:
        sub = df[df.event_type == clase].sort_values("dur")
        if sub.empty:
            continue
        # se elige el evento de duracion mediana: representativo de la clase
        idxs = [len(sub) // 2]
        for extra in range(1, args.n):
            j = (len(sub) // 2 + extra * max(1, len(sub) // (args.n + 1)))
            if j < len(sub):
                idxs.append(j)

        for pos, j in enumerate(idxs):
            ev = sub.iloc[j]
            base = f"{clase}_{pos}_ch{args.canal}"

            f1 = os.path.join(FIGURAS, f"validacion_{base}.png")
            figura_validacion(a, ev, args.canal, f1, catalogo=df)
            generadas.append(f1)

            f2 = os.path.join(FIGURAS, f"representacion_{base}.png")
            if figura_representacion(a, ev, args.canal, f2):
                generadas.append(f2)
            print(f"  {clase}: muestras {int(ev.idx_start)}-"
                  f"{int(ev.idx_end)} ({ev.dur:.1f} s)")

    print(f"\n{len(generadas)} figuras en {FIGURAS}")
    print("""
QUE MIRAR
  validacion_*.png    Compara a ojo el intervalo del especialista con lo
                      que marco cada detector. Si STA/LTA corta antes del
                      final del evento, aqui se ve directamente, y eso
                      explica el sesgo de duracion de -8,9 s.

  representacion_*.png  Es lo que consume el VGG16. Compara un IC (8 s)
                      con un TR (88 s): la misma red tiene que aceptar
                      ambos en una entrada de tamano fijo. Ahi se ve el
                      argumento para revisar la arquitectura.

PENDIENTE
  Los parametros del espectrograma (nperseg, noverlap, escala) deben
  copiarse del repositorio de Curilem antes de generar el conjunto de
  entrenamiento real. Los de aqui son solo para inspeccion visual.
""")


if __name__ == "__main__":
    main()
