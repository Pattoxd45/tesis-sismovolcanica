#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barrido_stalta.py
=================

Busca el punto de operacion de STA/LTA sobre la traza continua del NVChVC,
en vez de fijar parametros a ojo.

CORRIGE el defecto detectado en el diagnostico: la coincidencia entre
estaciones se evalua a nivel de INTERVALO con ventana de tolerancia, no
muestra a muestra. Con disparos de 2-3 s y eventos de 9-88 s, exigir
solapamiento exacto entre estaciones hundia el recall de 0,43 a 0,05.

Estrategia de costo: la funcion caracteristica (lo caro) se calcula una
sola vez por combinacion (canal, sta, lta) y se reutiliza para barrer
umbrales, fusion y coincidencia (lo barato).

Uso:
    python barrido_stalta.py                    # barrido completo
    python barrido_stalta.py --rapido           # rejilla reducida
    python barrido_stalta.py --canales 1 2 3
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_detectores import (Intervalo, metricas, CLASES, FS,
                                CORTES, GUARDA_S, NPY, CSV, SALIDA)

# =====================================================================
# REJILLA DE BUSQUEDA
# =====================================================================

REJILLA = dict(
    lta_s=[10.0, 30.0, 60.0],
    sta_s=[0.5, 1.0, 2.0],
    thr_on=[2.5, 3.0, 4.0],
    thr_off=[1.05, 1.2, 1.5],
    fusion_s=[5.0, 15.0],       # union de disparos dentro de un canal
    min_est=[1, 2],             # estaciones que deben coincidir
    tol_s=[5.0],                # tolerancia de coincidencia entre canales
)

REJILLA_RAPIDA = dict(
    lta_s=[10.0, 45.0],
    sta_s=[1.0],
    thr_on=[3.0],
    thr_off=[1.2],
    fusion_s=[15.0],
    min_est=[1, 2],
    tol_s=[5.0],
)

BANDA = (1.0, 20.0)
DUR_MIN_S = 3.0
IOU_MIN = 0.30


def segmentos(n):
    b = [0] + CORTES + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


# =====================================================================
# AGREGACION CORREGIDA
# =====================================================================

def fusionar_canal(trigs, gap):
    """Une disparos del MISMO canal separados por menos de gap muestras."""
    if not trigs:
        return []
    trigs = sorted(trigs)
    out = [list(trigs[0])]
    for s, e in trigs[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def coincidencia(por_canal, min_est, tol, dur_min, dur_max=40_000):
    """
    Agrupa intervalos de distintos canales que se solapan dentro de una
    tolerancia y emite una deteccion de red si participan >= min_est
    canales distintos.

    dur_max (en muestras, 400 s por defecto) corta el encadenamiento
    transitivo: si A toca a B y B toca a C, los tres caen en el mismo
    grupo, y en periodos activos la cadena llegaba a producir una unica
    "deteccion" de mas de una hora que absorbia eventos reales.

    por_canal : dict {canal: [(ini, fin), ...]}
    Devuelve  : lista de Intervalo (union temporal del grupo)
    """
    todos = [(s, e, ch) for ch, ivs in por_canal.items() for s, e in ivs]
    if not todos:
        return []
    todos.sort()

    grupos, actual = [], [todos[0]]
    fin_actual = todos[0][1]
    for s, e, ch in todos[1:]:
        if s <= fin_actual + tol and (max(fin_actual, e) -
                                      actual[0][0]) <= dur_max:
            actual.append((s, e, ch))
            fin_actual = max(fin_actual, e)
        else:
            grupos.append(actual)
            actual, fin_actual = [(s, e, ch)], e
    grupos.append(actual)

    salida = []
    for g in grupos:
        if len({ch for _, _, ch in g}) < min_est:
            continue
        ini = min(s for s, _, _ in g)
        fin = max(e for _, e, _ in g)
        if dur_min <= fin - ini <= dur_max:
            salida.append(Intervalo(int(ini), int(fin)))
    return salida


# =====================================================================
# BARRIDO
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canales", nargs="+", type=int, default=[1, 2, 3, 5, 6])
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--iou-min", type=float, default=IOU_MIN)
    args = ap.parse_args()

    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from obspy.signal.filter import bandpass

    rejilla = REJILLA_RAPIDA if args.rapido else REJILLA

    a = np.load(NPY, mmap_mode="r")
    df = pd.read_csv(CSV)
    n = a.shape[-1]

    valida = np.ones(n, dtype=bool)
    g = int(GUARDA_S * FS)
    for c in CORTES:
        valida[max(0, c - g):min(n, c + g)] = False

    catalogo = [Intervalo(int(r.idx_start), int(r.idx_end), r.event_type)
                for r in df.itertuples()
                if valida[int(r.idx_start)] and valida[int(r.idx_end) - 1]]
    horas = float(valida.sum()) / FS / 3600.0

    print(f"Catalogo: {len(catalogo)} eventos | {horas:.2f} h validas")
    print(f"Canales : {args.canales}")

    # --- filtrado una sola vez por canal ---------------------------
    print("\nFiltrando canales ...", flush=True)
    filtrado = {}
    for ch in args.canales:
        partes = []
        for ini, fin in segmentos(n):
            x = np.asarray(a[ch, ini:fin], dtype=np.float64)
            if np.allclose(x, 0):
                partes.append((ini, fin, None))
                continue
            partes.append((ini, fin, bandpass(x, BANDA[0], BANDA[1], df=FS,
                                              corners=4, zerophase=True)))
        filtrado[ch] = partes
        print(f"  canal {ch} listo", flush=True)

    combos_cft = list(itertools.product(rejilla["lta_s"], rejilla["sta_s"]))
    combos_thr = list(itertools.product(
        rejilla["thr_on"], rejilla["thr_off"], rejilla["fusion_s"],
        rejilla["min_est"], rejilla["tol_s"]))
    print(f"\n{len(combos_cft)} funciones caracteristicas x "
          f"{len(combos_thr)} umbrales = "
          f"{len(combos_cft) * len(combos_thr)} combinaciones\n")

    filas = []
    for lta_s, sta_s in combos_cft:
        if sta_s >= lta_s:
            continue
        print(f">>> sta={sta_s}s lta={lta_s}s  (calculando cft)", flush=True)

        cft_por_canal = {}
        for ch in args.canales:
            trozos = []
            for ini, fin, xf in filtrado[ch]:
                if xf is None:
                    trozos.append((ini, None))
                    continue
                c = classic_sta_lta(xf, int(sta_s * FS), int(lta_s * FS))
                trozos.append((ini, np.nan_to_num(c, nan=0.0, posinf=0.0,
                                                  neginf=0.0)))
            cft_por_canal[ch] = trozos

        for thr_on, thr_off, fus_s, k, tol_s in combos_thr:
            if thr_off >= thr_on:
                continue
            por_canal = {}
            for ch in args.canales:
                trigs = []
                for ini, c in cft_por_canal[ch]:
                    if c is None:
                        continue
                    for t0, t1 in trigger_onset(c, thr_on, thr_off):
                        trigs.append((ini + int(t0), ini + int(t1)))
                por_canal[ch] = fusionar_canal(trigs, int(fus_s * FS))

            det = coincidencia(por_canal, k, int(tol_s * FS),
                               int(DUR_MIN_S * FS))
            det = [d for d in det if valida[d.ini] and valida[d.fin - 1]]

            res, *_ = metricas(catalogo, det, horas, args.iou_min)
            filas.append(dict(
                sta_s=sta_s, lta_s=lta_s, thr_on=thr_on, thr_off=thr_off,
                fusion_s=fus_s, min_est=k, tol_s=tol_s,
                n_det=res["n_detecciones"], recall=res["recall"],
                precision=res["precision"], f1=res["f1"],
                fp_h=res["fp_por_hora"], iou=res["iou_medio"],
                onset_mae=res["onset_mae_s"],
                dur_sesgo=res["duracion_sesgo_s"],
                **{f"rec_{c}": res["por_clase"].get(c, {}).get("recall",
                                                               float("nan"))
                   for c in CLASES}))

    tabla = pd.DataFrame(filas).sort_values("f1", ascending=False)

    os.makedirs(SALIDA, exist_ok=True)
    destino = f"{SALIDA}/barrido_stalta.csv"
    tabla.to_csv(destino, index=False)

    cols = ["sta_s", "lta_s", "thr_on", "thr_off", "fusion_s", "min_est",
            "n_det", "recall", "precision", "f1", "fp_h", "iou", "onset_mae"]
    print("\n" + "=" * 78)
    print("  MEJORES 15 COMBINACIONES (por F1)")
    print("=" * 78)
    print(tabla[cols].head(15).to_string(index=False,
                                         float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("  MEJOR RECALL CON FP/h < 10")
    print("=" * 78)
    lim = tabla[tabla.fp_h < 10].sort_values("recall", ascending=False)
    if len(lim):
        print(lim[cols].head(10).to_string(index=False,
                                           float_format=lambda v: f"{v:.3f}"))
    else:
        print("  Ninguna combinacion baja de 10 FP/h.")

    print("\n" + "=" * 78)
    print("  RECALL POR CLASE EN EL MEJOR PUNTO (F1)")
    print("=" * 78)
    mejor = tabla.iloc[0]
    for c in CLASES:
        print(f"  {c}: {mejor[f'rec_{c}']:.3f}")
    print(f"\n  (TR bajo es esperable: la LTA se adapta a la senal "
          f"sostenida del tremor)")
    print(f"\nTabla completa en {destino}")


if __name__ == "__main__":
    main()
