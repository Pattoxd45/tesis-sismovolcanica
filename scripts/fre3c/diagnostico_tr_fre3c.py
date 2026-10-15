#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nivel de azar de la sensibilidad y diagnostico de los TR perdidos.

Uso: python diagnostico_tr_fre3c.py [--umbral 0.10] [--filtro f1_15]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fre3c_lib as lib

FS = lib.FS
TOL = int(round(lib.TOL_S * FS))
VENTANA_AZAR_S = 2 * lib.TOL_S          # ventana de acierto: +-5 s
TARDIOS_S = [5, 10, 20, 30, 60]          # ventanas [inicio - 5 s, inicio + T]
FLUJOS_AZAR = ["PN-PS", "PN-P", "PN-S", "EQT-D"]
UMBRALES_AZAR = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def nivel_azar(sc, st, filtro):
    v = sc[(sc.clase == "volcanico") & (sc.tramo.astype(str) == "todos") & (sc.filtro == filtro)]
    t = st[(st.tramo.astype(str) == "todos") & (st.filtro == filtro)]
    d = v.merge(t, on=["flujo", "filtro", "umbral", "tramo"])
    d = d[d.flujo.isin(FLUJOS_AZAR) & d.umbral.round(2).isin(UMBRALES_AZAR)].copy()
    lam = d.fp_h / 3600.0
    d["p_azar"] = 1 - np.exp(-lam * VENTANA_AZAR_S)
    d["sens_corregida"] = np.maximum(0.0, (d.sens - d.p_azar) / (1 - d.p_azar))
    # aciertos atribuibles al azar = n * (sens - sens_corregida)
    d["aciertos_azar_esperados"] = d.p_azar * (d.n - d.k) / (1 - d.p_azar)
    return d[["flujo", "umbral", "k", "n", "sens", "fp_h", "p_azar", "sens_corregida",
              "aciertos_azar_esperados"]].sort_values(["flujo", "umbral"])


def onsets_pn_ps(curvas, umbral):
    partes = []
    for et in ("P", "S"):
        _, _, pico, _ = lib.disparos_por_tramo(curvas[et], umbral)
        partes.append(pico)
    return lib.agrupar_anclado(np.concatenate(partes), TOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.10)
    ap.add_argument("--filtro", default="f1_15", choices=list(lib.FILTROS))
    args = ap.parse_args()

    sc = pd.read_csv(os.path.join(lib.DIR_SALIDA, "sens_clase.csv"))
    st = pd.read_csv(os.path.join(lib.DIR_SALIDA, "sens_totales.csv"))
    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    valida = np.load(lib.ruta_datos(args.filtro))["valida"]
    z = np.load(lib.ruta_anot("phasenet", args.filtro))
    curvas = {k: np.nan_to_num(z[k]) for k in ("P", "S")}

    print(f"1. NIVEL DE AZAR (sensibilidad volcanica, filtro {args.filtro})")
    print("   p_azar = probabilidad de que una falsa alarma caiga por casualidad en los 10 s")
    print("   de un evento; sens_corregida = (sens - p_azar) / (1 - p_azar).")
    az = nivel_azar(sc, st, args.filtro)
    print(az.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    az.to_csv(os.path.join(lib.DIR_SALIDA, "nivel_azar.csv"), index=False)

    inc = eventos[eventos.incluido].reset_index(drop=True)
    o = onsets_pn_ps(curvas, args.umbral)
    o = o[valida[o]]
    tot = st[(st.flujo == "PN-PS") & (st.filtro == args.filtro) & np.isclose(st.umbral, args.umbral)
             & (st.tramo.astype(str) == "todos")]
    lam = float(tot.fp_h.iloc[0]) / 3600.0 if len(tot) else float("nan")

    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    res = lib.evaluar(o, inc, (con_senal.ini - TOL).to_numpy(), con_senal.fin.to_numpy(), valida)
    otros = np.sort(eventos.ini.to_numpy())

    filas = []
    for i, r in inc.iterrows():
        if r.clase != "TR":
            continue
        a, b = int(r.ini), int(r.fin)
        lo, hi = max(0, a - TOL), a + TOL + 1
        dentro = o[(o >= a - TOL) & (o <= b)]
        vecinos = otros[otros != a]
        dist = np.min(np.abs(vecinos - a)) / FS if vecinos.size else np.nan
        filas.append(dict(
            id=int(r.id), tramo=int(r.tramo), dur_s=round(r.dur_s, 1),
            acierta=bool(res["asignado"][i] >= 0),
            dt_s=round(float(res["dt"][i]), 2) if res["asignado"][i] >= 0 else np.nan,
            maxP_inicio=round(float(curvas["P"][lo:hi].max()), 3),
            maxS_inicio=round(float(curvas["S"][lo:hi].max()), 3),
            maxP_evento=round(float(curvas["P"][a:b].max()), 3),
            primera_dentro_s=round((dentro.min() - a) / FS, 1) if dentro.size else np.nan,
            n_dentro=int(dentro.size),
            otro_evento_mas_cercano_s=round(float(dist), 1)))
    tr = pd.DataFrame(filas)
    tr.to_csv(os.path.join(lib.DIR_SALIDA, "diagnostico_tr.csv"), index=False)

    print(f"\n2. TR CON PHASENET P Y S (umbral {args.umbral:.2f}, filtro {args.filtro}): "
          f"{int(tr.acierta.sum())}/{len(tr)} aciertan")
    print("   maxP/maxS_inicio: maximo de la curva en +-5 s del inicio (comparar con el umbral)")
    print("   primera_dentro_s: primera deteccion entre inicio - 5 s y el fin, medida desde el inicio")
    print("\n   TR PERDIDOS:")
    if (~tr.acierta).any():
        print(tr[~tr.acierta].drop(columns=["acierta", "dt_s"]).to_string(index=False))
    else:
        print("   ninguno")
    print("\n   TR DETECTADOS (para comparar), medianas:")
    det = tr[tr.acierta]
    if len(det):
        print(f"   dur {det.dur_s.median():.1f} s | maxP_inicio {det.maxP_inicio.median():.3f} | "
              f"maxS_inicio {det.maxS_inicio.median():.3f} | dt {det.dt_s.median():.2f} s")
    per = tr[~tr.acierta]
    if len(per):
        print(f"   perdidos: dur {per.dur_s.median():.1f} s | maxP_inicio "
              f"{per.maxP_inicio.median():.3f} | maxS_inicio {per.maxS_inicio.median():.3f}")

    print("\n3. SI SE ACEPTARA UN AVISO TARDIO PARA TR (ventana [inicio - 5 s, inicio + T]):")
    print("   cuenta como detectado si hay alguna deteccion en la ventana (sin asignacion uno a uno);")
    print("   p_azar = probabilidad de que una falsa alarma caiga en esa ventana por casualidad.")
    print(f"   {'T (s)':>6} {'TR detect.':>11} {'sens':>6} {'p_azar':>7} {'sens corregida':>15}")
    for T in TARDIOS_S:
        k = 0
        for _, r in inc[inc.clase == "TR"].iterrows():
            a = int(r.ini)
            k += bool(np.any((o >= a - TOL) & (o <= a + int(T * FS))))
        n = int((inc.clase == "TR").sum())
        p = 1 - np.exp(-lam * (T + lib.TOL_S))
        s = k / n
        print(f"   {T:>6} {k:>6}/{n:<4} {s:>6.3f} {p:>7.3f} {max(0.0, (s - p) / (1 - p)):>15.3f}")
    print(f"\nTablas en {lib.DIR_SALIDA}/nivel_azar.csv y diagnostico_tr.csv")


if __name__ == "__main__":
    main()
