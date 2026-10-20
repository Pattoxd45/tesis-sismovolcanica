#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico de los TR que PhaseNet (P y S) no detecta.

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

    print(f"\nTabla en {lib.DIR_SALIDA}/diagnostico_tr.csv")


if __name__ == "__main__":
    main()
