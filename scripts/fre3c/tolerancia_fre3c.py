#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensibilidad por clase con distintas tolerancias de tiempo, igual para PhaseNet,
EQTransformer y STA/LTA, retraso del primer aviso de cada evento y umbral de las redes
con 5 y con 10 s.

Uso: python tolerancia_fre3c.py [--umbral 0.10] [--filtro f1_15]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
import fre3c_lib as lib

FS = lib.FS
SIMETRICAS_S = [1, 2, 3, 5, 7, 10, 15, 20, 30]
TARDIAS_S = [5, 7, 10, 15, 20, 30, 60]
ANTES_TARDIA_S = 5.0     # en la prueba tardia solo se alarga el lado posterior al inicio
DETECTORES = ["PhaseNet P y S", "EQTransformer deteccion", "EQTransformer P y S", "STA/LTA"]
FLUJOS_UMBRAL = {"PhaseNet P y S": ("PN-PS", "phasenet"), "EQTransformer deteccion": ("EQT-D", "eqtransformer"),
                 "EQTransformer P y S": ("EQT-PS", "eqtransformer")}
TOL_INFORME_S = [5, 10]
METAS = [0.80, 0.90, 0.95]


def onsets_detectores(umbral, filtro, valida):
    import evaluar_detectores as ed
    import recorte_fre3c as rc
    import sensibilidad_fre3c as sf

    pn = dict(np.load(lib.ruta_anot("phasenet", filtro)))
    eqt = dict(np.load(lib.ruta_anot("eqtransformer", filtro)))
    out = {"PhaseNet P y S": sf.onsets_de("PN-PS", pn, umbral, {}),
           "EQTransformer deteccion": sf.onsets_de("EQT-D", eqt, umbral, {}),
           "EQTransformer P y S": sf.onsets_de("EQT-PS", eqt, umbral, {})}
    z = np.load(lib.ruta_datos(filtro))["Z"]
    ivs = rc.recortes_stalta(rc.cft_stalta(z), valida, ed.STALTA["thr_on"])
    out["STA/LTA"] = np.asarray([a for a, _ in ivs], dtype=np.int64)
    return out


def evaluar_ventana(onsets, inc, zona_ini, zona_fin, valida, antes_s, despues_s):
    """Como lib.evaluar, con ventana [inicio - antes_s, inicio + despues_s]; un aviso por evento."""
    antes, despues = int(round(antes_s * FS)), int(round(despues_s * FS))
    o = np.unique(np.asarray(onsets, dtype=np.int64))
    o = o[(o >= 0) & (o < valida.size)]
    o = o[valida[o]]
    inicios = inc["ini"].to_numpy(dtype=np.int64)
    cand = []
    for i, e in enumerate(inicios):
        a = int(np.searchsorted(o, e - antes, side="left"))
        b = int(np.searchsorted(o, e + despues, side="right"))
        cand += [(abs(int(o[j]) - int(e)), i, j) for j in range(a, b)]
    cand.sort()
    asignado = np.full(inicios.size, -1, dtype=np.int64)
    usado = np.zeros(o.size, dtype=bool)
    for _, i, j in cand:
        if asignado[i] < 0 and not usado[j]:
            asignado[i], usado[j] = j, True
    dt = np.full(inicios.size, np.nan)
    ok = asignado >= 0
    dt[ok] = (o[asignado[ok]] - inicios[ok]) / FS
    cat = np.full(o.size, "externa", dtype=object)
    # un aviso dentro de un evento o de su ventana previa no es falsa alarma
    cat[lib.dentro_de_intervalos(o, zona_ini - antes, zona_fin)] = "interna"
    cat[usado] = "acierto"
    return dict(onsets=o, asignado=asignado, dt=dt, cat=cat)


def fila(detector, tipo, antes_s, despues_s, res, inc, horas):
    fc, ft = lib.resumir(res, inc, horas)
    c = {f["clase"]: f for f in fc}
    return dict(detector=detector, tipo=tipo, antes_s=antes_s, despues_s=despues_s,
                **{f"k_{k}": c[k]["k"] for k in lib.CLASES},
                **{f"n_{k}": c[k]["n"] for k in lib.CLASES},
                k_volc=c["volcanico"]["k"], n_volc=c["volcanico"]["n"],
                sens_volc=c["volcanico"]["sens"], fp_h=ft["fp_h"], fdr=ft["fdr"])


def retrasos(onsets, inc, valida, detector):
    """Primer aviso entre inicio - 5 s y el fin de cada evento, medido desde el inicio."""
    tol = int(round(lib.TOL_S * FS))
    o = np.unique(np.asarray(onsets, dtype=np.int64))
    o = o[(o >= 0) & (o < valida.size)]
    o = np.sort(o[valida[o]])
    filas = []
    for r in inc.itertuples():
        k = int(np.searchsorted(o, int(r.ini) - tol, side="left"))
        primero = o[k] if k < o.size and o[k] <= int(r.fin) else None
        filas.append(dict(detector=detector, id=int(r.id), clase=r.clase, dur_s=r.dur_s,
                          retraso_s=(primero - int(r.ini)) / FS if primero is not None else np.nan))
    return filas


def imprimir(df, tipo, valores, col):
    print(f"   {'detector':<25} {'T (s)':>5} {'VT':>7} {'LP':>7} {'TR':>7} {'volcanicos':>10} "
          f"{'sens':>6} {'FP/h':>6}")
    for det in DETECTORES:
        for v in valores:
            r = df[(df.detector == det) & (df.tipo == tipo) & np.isclose(df[col], v)].iloc[0]
            kn = lambda c: f"{r[f'k_{c}']}/{r[f'n_{c}']}"
            print(f"   {det:<25} {v:>5g} {kn('VT'):>7} {kn('LP'):>7} {kn('TR'):>7} "
                  f"{r.k_volc:>5}/{r.n_volc:<4} {r.sens_volc:>6.3f} {r.fp_h:>6.1f}")
        print()


def main():
    import sensibilidad_fre3c as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.10)
    ap.add_argument("--filtro", default="f1_15", choices=list(lib.FILTROS))
    args = ap.parse_args()

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    valida = np.load(lib.ruta_datos("f1_15"))["valida"]
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zona_ini, zona_fin = con_senal.ini.to_numpy(), con_senal.fin.to_numpy()
    horas = valida.sum() / FS / 3600
    ons = onsets_detectores(args.umbral, args.filtro, valida)

    filas, filas_r = [], []
    for det in DETECTORES:
        for t in SIMETRICAS_S:
            res = evaluar_ventana(ons[det], inc, zona_ini, zona_fin, valida, t, t)
            filas.append(fila(det, "simetrica", t, t, res, inc, horas))
        for t in TARDIAS_S:
            res = evaluar_ventana(ons[det], inc, zona_ini, zona_fin, valida, ANTES_TARDIA_S, t)
            filas.append(fila(det, "tardia", ANTES_TARDIA_S, t, res, inc, horas))
        filas_r += retrasos(ons[det], inc, valida, det)
    df, dr = pd.DataFrame(filas), pd.DataFrame(filas_r)

    print(f"TOLERANCIA DE TIEMPO (redes con umbral {args.umbral:.2f}, filtro {args.filtro}; "
          "STA/LTA con la configuracion de la Etapa 2)")
    print("  un aviso por evento; FP/h = avisos fuera de todo evento y de su ventana previa")
    tol = int(round(lib.TOL_S * FS))
    for det in DETECTORES:
        base = lib.evaluar(ons[det], inc, (con_senal.ini - tol).to_numpy(), zona_fin, valida)
        fc, ft = lib.resumir(base, inc, horas)
        r = df[(df.detector == det) & (df.tipo == "simetrica") & np.isclose(df.antes_s, lib.TOL_S)].iloc[0]
        igual = {f["clase"]: f["k"] for f in fc}["volcanico"] == r.k_volc and np.isclose(ft["fp_h"], r.fp_h)
        print(f"  comprobacion {det}: +-5 s igual a la evaluacion normal: {'SI' if igual else 'NO'}")

    print("\n1. TOLERANCIA SIMETRICA: aviso a T s o menos del inicio, antes o despues")
    imprimir(df, "simetrica", SIMETRICAS_S, "antes_s")
    print(f"2. TOLERANCIA TARDIA: aviso entre {ANTES_TARDIA_S:g} s antes y T s despues del inicio")
    imprimir(df, "tardia", TARDIAS_S, "despues_s")

    print("3. RETRASO DEL PRIMER AVISO (entre inicio - 5 s y el fin del evento), en s")
    print(f"   {'detector':<25} {'clase':>5} {'con aviso':>9} {'mediana':>8} {'p75':>6} {'p90':>6} "
          f"{'> 5 s':>6} {'sin aviso':>9}")
    for det in DETECTORES:
        for cl in lib.VOLCANICAS:
            g = dr[(dr.detector == det) & (dr.clase == cl)]
            v = g.retraso_s.dropna()
            if v.empty:
                print(f"   {det:<25} {cl:>5} {0:>9} {'-':>8} {'-':>6} {'-':>6} {0:>6} {len(g):>9}")
                continue
            print(f"   {det:<25} {cl:>5} {len(v):>9} {v.median():>8.1f} {v.quantile(0.75):>6.1f} "
                  f"{v.quantile(0.90):>6.1f} {int((v > lib.TOL_S).sum()):>6} {len(g) - len(v):>9}")
        print()

    # 5 y 10 s: los dos valores que se informan; el umbral de cada red se barre igual que en sensibilidad_fre3c.py
    print("4. UMBRAL DE LAS REDES CON 5 Y CON 10 s (simetrica; P y S con el mismo umbral)")
    filas_u = []
    for det, (flujo, modelo) in FLUJOS_UMBRAL.items():
        curvas = dict(np.load(lib.ruta_anot(modelo, args.filtro)))
        cache = {}
        for u in lib.UMBRALES:
            o = sf.onsets_de(flujo, curvas, u, cache)
            for t in TOL_INFORME_S:
                res = evaluar_ventana(o, inc, zona_ini, zona_fin, valida, t, t)
                filas_u.append(dict(fila(det, "umbral", t, t, res, inc, horas), umbral=u))
    du = pd.DataFrame(filas_u)
    for t in TOL_INFORME_S:
        tol_t = int(round(t * FS))
        base = lib.evaluar(ons["PhaseNet P y S"], inc, zona_ini - tol_t, zona_fin, valida, t)
        fc, ft = lib.resumir(base, inc, horas)
        r = du[(du.detector == "PhaseNet P y S") & np.isclose(du.umbral, args.umbral) & np.isclose(du.antes_s, t)]
        igual = (not r.empty and {f["clase"]: f["k"] for f in fc}["volcanico"] == r.iloc[0].k_volc
                 and np.isclose(ft["fp_h"], r.iloc[0].fp_h))
        print(f"   comprobacion PhaseNet +-{t:g} s, umbral {args.umbral:.2f}: igual a lib.evaluar: {'SI' if igual else 'NO'}")
    print(f"   {'detector':<25} {'umbral':>6} " + " ".join(f"{f'{t:g} s: sens':>11} {'FP/h':>6}" for t in TOL_INFORME_S))
    for det in FLUJOS_UMBRAL:
        for u in lib.UMBRALES:
            celdas = []
            for t in TOL_INFORME_S:
                r = du[(du.detector == det) & np.isclose(du.umbral, u) & np.isclose(du.antes_s, t)].iloc[0]
                celdas.append(f"{r.sens_volc:>11.3f} {r.fp_h:>6.1f}")
            print(f"   {det:<25} {u:>6.2f} " + " ".join(celdas))
        print()
    print("   Menos FP/h que alcanza cada meta (VT, LP, TR detectados entre parentesis):")
    for det in FLUJOS_UMBRAL:
        for t in TOL_INFORME_S:
            g = du[(du.detector == det) & np.isclose(du.antes_s, t)]
            partes = []
            for meta in METAS:
                ok = g[g.sens_volc >= meta]
                if ok.empty:
                    partes.append(f"{meta:.2f}: no")
                    continue
                r = ok.sort_values(["fp_h", "umbral"], ascending=[True, False]).iloc[0]
                partes.append(f"{meta:.2f}: u {r.umbral:.2f}, {r.sens_volc:.3f}, {r.fp_h:.1f} FP/h "
                              f"({r.k_VT}/{r.k_LP}/{r.k_TR})")
            mx = g.sort_values(["sens_volc", "fp_h"], ascending=[False, True]).iloc[0]
            print(f"   {det:<25} {t:>3g} s | " + " | ".join(partes) +
                  f" | max {mx.sens_volc:.3f} con u {mx.umbral:.2f}, {mx.fp_h:.1f} FP/h")
    du.to_csv(os.path.join(lib.DIR_SALIDA, "tolerancia_umbral.csv"), index=False)

    df.to_csv(os.path.join(lib.DIR_SALIDA, "tolerancias.csv"), index=False)
    dr.to_csv(os.path.join(lib.DIR_SALIDA, "retrasos.csv"), index=False)
    print(f"Tablas en {lib.DIR_SALIDA}/tolerancias.csv, retrasos.csv y tolerancia_umbral.csv")


if __name__ == "__main__":
    main()
