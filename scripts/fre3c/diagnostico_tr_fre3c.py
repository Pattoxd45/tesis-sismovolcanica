#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico de los TR que PhaseNet (P y S) no detecta y prueba de aceptar avisos tardios.

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
TARDIOS_S = [5, 10, 20, 30, 60]  # el tremor empieza de a poco y PhaseNet avisa tarde


def onsets_pn_ps(curvas, umbral):
    partes = [lib.disparos_por_tramo(curvas[et], umbral)[2] for et in ("P", "S")]
    return lib.agrupar_anclado(np.concatenate(partes), TOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.10)
    ap.add_argument("--filtro", default="f1_15", choices=list(lib.FILTROS))
    args = ap.parse_args()

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    valida = np.load(lib.ruta_datos(args.filtro))["valida"]
    z = np.load(lib.ruta_anot("phasenet", args.filtro))
    curvas = {k: np.nan_to_num(z[k]) for k in ("P", "S")}

    inc = eventos[eventos.incluido].reset_index(drop=True)
    o = onsets_pn_ps(curvas, args.umbral)
    o = o[valida[o]]
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

    print(f"1. TR CON PHASENET P Y S (umbral {args.umbral:.2f}, filtro {args.filtro}): "
          f"{int(tr.acierta.sum())}/{len(tr)} detectados")
    print("   maxP/maxS_inicio: maximo de la curva a 5 s o menos del inicio (comparar con el umbral)")
    print("   primera_dentro_s: primer aviso entre inicio - 5 s y el fin, medido desde el inicio")
    print("\n   TR PERDIDOS:")
    if (~tr.acierta).any():
        print(tr[~tr.acierta].drop(columns=["acierta", "dt_s"]).to_string(index=False))
    else:
        print("   ninguno")
    det = tr[tr.acierta]
    if len(det):
        print(f"\n   TR detectados, medianas: dur {det.dur_s.median():.1f} s | maxP_inicio "
              f"{det.maxP_inicio.median():.3f} | maxS_inicio {det.maxS_inicio.median():.3f} | "
              f"dt {det.dt_s.median():.2f} s")
    per = tr[~tr.acierta]
    if len(per):
        print(f"   TR perdidos, medianas: dur {per.dur_s.median():.1f} s | maxP_inicio "
              f"{per.maxP_inicio.median():.3f} | maxS_inicio {per.maxS_inicio.median():.3f}")

    # VT y LP quedan como en la evaluacion normal: 5 s y un aviso por evento
    volc = inc.clase.isin(lib.VOLCANICAS).to_numpy()
    k_vt_lp = int(((res["asignado"] >= 0) & inc.clase.isin(["VT", "LP"]).to_numpy()).sum())
    n_volc = int(volc.sum())
    n_tr = int((inc.clase == "TR").sum())

    print("\n2. SI PARA TR SE ACEPTARA UN AVISO TARDIO (entre inicio - 5 s e inicio + T):")
    print("   TR cuenta si hay algun aviso en esa ventana; VT y LP se cuentan como siempre.")
    print(f"   {'T (s)':>6} {'TR':>8} {'sens TR':>8} {'volcanicos':>11} {'sens volc':>10}")
    tol_tardia = []
    for T in TARDIOS_S:
        k_tr = 0
        for r in inc[inc.clase == "TR"].itertuples():
            a = int(r.ini)
            k_tr += bool(np.any((o >= a - TOL) & (o <= a + int(T * FS))))
        k_v = k_vt_lp + k_tr
        print(f"   {T:>6} {k_tr:>4}/{n_tr:<3} {k_tr / n_tr:>8.3f} {k_v:>6}/{n_volc:<4} "
              f"{k_v / n_volc:>10.3f}")
        tol_tardia.append(dict(T_s=T, k_tr=k_tr, n_tr=n_tr, sens_tr=k_tr / n_tr,
                               k_volc=k_v, n_volc=n_volc, sens_volc=k_v / n_volc))
    pd.DataFrame(tol_tardia).to_csv(os.path.join(lib.DIR_SALIDA, "tolerancia_tr.csv"), index=False)
    print(f"\nTablas en {lib.DIR_SALIDA}/diagnostico_tr.csv y tolerancia_tr.csv")


if __name__ == "__main__":
    main()
