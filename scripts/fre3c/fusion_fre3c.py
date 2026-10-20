#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion de PhaseNet (P y S) y EQTransformer (inicio de la deteccion) en el nivel 1.
Todos los arribos van a una sola lista; cada grupo abre con el arribo mas temprano,
suma los que caen a 5 s o menos de ese primero, sin encadenar, y se representa con
el mas temprano. Es la misma agrupacion que une P y S en PN-PS.

Uso: python fusion_fre3c.py [--filtro-eqt f1_45]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fre3c_lib as lib
import sensibilidad_fre3c as sf

FS = lib.FS
TOL = int(round(lib.TOL_S * FS))
UMBRALES_PN = [0.05, 0.10, 0.30]
UMBRALES_EQT = [0.10, 0.30, 0.50]


def fusionar(*listas):
    t = np.concatenate([np.asarray(x, dtype=np.int64) for x in listas])
    return lib.agrupar_anclado(t, TOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtro-pn", default="f1_15", choices=list(lib.FILTROS))
    ap.add_argument("--filtro-eqt", default="f1_45", choices=list(lib.FILTROS))
    args = ap.parse_args()

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    valida = np.load(lib.ruta_datos("f1_15"))["valida"]
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zi, zf = (con_senal.ini - TOL).to_numpy(), con_senal.fin.to_numpy()
    horas = valida.sum() / FS / 3600
    pn = dict(np.load(lib.ruta_anot("phasenet", args.filtro_pn)))
    eqt = dict(np.load(lib.ruta_anot("eqtransformer", args.filtro_eqt)))
    cpn, ceqt = {}, {}

    def medir(o):
        res = lib.evaluar(o, inc, zi, zf, valida)
        fc, tot = lib.resumir(res, inc, horas)
        return res, {f["clase"]: f for f in fc}, tot

    volc = inc.clase.isin(lib.VOLCANICAS).to_numpy()
    filas = []
    print(f"FUSION PHASENET P Y S ({args.filtro_pn}) + EQTRANSFORMER DETECCION ({args.filtro_eqt}), "
          "NIVEL 1, tolerancia 5 s")
    print("  k/n por clase; VOLC = VT+LP+TR; FP/h fuera de todo evento; FDR = FP/(aciertos volc + FP)")
    print("  nuevos: volcanicos que acierta la fusion y no PhaseNet solo, al mismo umbral\n")
    print(f"  {'PN':>5} {'EQT':>5} {'VT':>6} {'LP':>6} {'TR':>6} {'VOLC':>6} {'AV':>4} {'IC':>6} "
          f"{'FP/h':>6} {'FDR':>6} | {'PN solo':>7} {'FP/h':>6} | {'nuevos':>6} {'perdidos':>8}")
    for upn in UMBRALES_PN:
        o_pn = sf.onsets_de("PN-PS", pn, upn, cpn)
        r_pn, c_pn, t_pn = medir(o_pn)
        for ueq in UMBRALES_EQT:
            o_eq = sf.onsets_de("EQT-D", eqt, ueq, ceqt)
            r_f, c_f, t_f = medir(fusionar(o_pn, o_eq))
            a_pn = (r_pn["asignado"] >= 0) & volc
            a_f = (r_f["asignado"] >= 0) & volc
            nuevos = inc[a_f & ~a_pn]
            perdidos = inc[a_pn & ~a_f]
            kn = lambda c: f"{c_f[c]['k']}/{c_f[c]['n']}"
            print(f"  {upn:>5.2f} {ueq:>5.2f} {kn('VT'):>6} {kn('LP'):>6} {kn('TR'):>6} "
                  f"{c_f['volcanico']['sens']:>6.3f} {kn('AV'):>4} {kn('IC'):>6} {t_f['fp_h']:>6.1f} "
                  f"{t_f['fdr']:>6.3f} | {c_pn['volcanico']['sens']:>7.3f} {t_pn['fp_h']:>6.1f} | "
                  f"{len(nuevos):>6} {len(perdidos):>8}")
            filas.append(dict(umbral_pn=upn, umbral_eqt=ueq,
                              **{f"k_{c}": c_f[c]["k"] for c in lib.CLASES},
                              sens_volc=c_f["volcanico"]["sens"], fp_h=t_f["fp_h"], fdr=t_f["fdr"],
                              sens_volc_pn=c_pn["volcanico"]["sens"], fp_h_pn=t_pn["fp_h"],
                              nuevos=";".join(f"{r.id}:{r.clase}" for r in nuevos.itertuples()),
                              perdidos=";".join(f"{r.id}:{r.clase}" for r in perdidos.itertuples())))
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(lib.DIR_SALIDA, "fusion_nivel1.csv"), index=False)
    print("\n  perdidos: volcanicos que PhaseNet solo acierta y la fusion no (el grupo queda")
    print("  representado por un arribo de EQTransformer a mas de 5 s del inicio)")
    x = df[np.isclose(df.umbral_pn, 0.10) & np.isclose(df.umbral_eqt, 0.30)].iloc[0]
    print(f"\n  A 0,10 / 0,30, nuevos: {x.nuevos or '-'} | perdidos: {x.perdidos or '-'}")
    print(f"\nTabla en {lib.DIR_SALIDA}/fusion_nivel1.csv")


if __name__ == "__main__":
    main()
