#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grilla de los parametros de annotate() de SeisBench (overlap, blinding, stacking) y de los
umbrales, con deteccion por interseccion, para PhaseNet y EQTransformer. Vuelve a anotar
FRE con cada combinacion y verifica la eleccion rotando los tramos.

Uso: python grilla_fre3c.py [--modelos phasenet eqtransformer]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fre3c_lib as lib
import anotar_fre3c as an
import interseccion_fre3c as ix

FS = lib.FS
FILTRO = "f1_15"
# overlap en muestras (ventana de 3001 en PhaseNet y 6000 en EQTransformer); blinding por lado
GRILLA = {
    "phasenet": dict(overlap=[500, 1000, 1500, 2000, 2500], blinding=[0, 100, 250, 500],
                     stacking=["avg", "max"]),
    "eqtransformer": dict(overlap=[1000, 2000, 3000, 4000, 5000], blinding=[0, 250, 500, 1000],
                          stacking=["max", "avg"]),
}
FLUJOS = {"phasenet": ["PhaseNet P y S"], "eqtransformer": ["EQTransformer P y S", "EQTransformer deteccion"]}
CLAVES = ["overlap", "blinding", "stacking", "P_threshold", "S_threshold", "detection_threshold"]


def por_defecto(m):
    """Valor efectivo de annotate(): default_args de los pesos o, si no estan, el de la clase."""
    out = {}
    for k in ("overlap", "blinding", "stacking"):
        v = m.default_args.get(k, m._annotate_args[k][1])
        out[k] = tuple(int(x) for x in v) if k == "blinding" else v
    return out


def combinaciones(modelo, dflt):
    g = GRILLA[modelo]
    b0 = dflt["blinding"]
    if b0[0] != b0[1]:
        raise SystemExit(f"DETENIDO: blinding por defecto asimetrico {b0}; la grilla supone igual a cada lado.")
    ov = sorted(set(g["overlap"]) | {int(dflt["overlap"])})
    bl = sorted(set(g["blinding"]) | {b0[0]})
    st = list(dict.fromkeys(g["stacking"] + [dflt["stacking"]]))
    dentro, fuera = [], []
    for o in ov:
        for b in bl:
            for s in st:
                # con overlap < 2*blinding quedan tramos de la traza sin prediccion
                (dentro if o >= 2 * b else fuera).append((o, b, s))
    return dentro, fuera


def anotar(m, datos, o, b, s):
    curvas = {}
    for a, z in lib.LIMITES_TRAMO:
        ann = m.annotate(an.stream_tramo(datos, a, z), overlap=o, blinding=(b, b), stacking=s)
        an.volcar(ann, curvas, a, z)
    return curvas


def barrer_umbrales(curvas, flujo, inc, zona, valida, horas, base):
    cache = {}
    if flujo.endswith("deteccion"):
        combos = [(np.nan, np.nan, u) for u in lib.UMBRALES]
    else:
        combos = [(up, us, np.nan) for up in lib.UMBRALES for us in lib.UMBRALES]
    filas = []
    for up, us, ud in combos:
        if flujo.endswith("deteccion"):
            pi, pf = ix.intervalos_fase(curvas, "Detection", ud, cache)
        else:
            pi, pf = ix.intervalos_ps(curvas, up, us, cache)
        r = ix.evaluar_interseccion(pi, pf, inc, zona, valida)
        filas += ix.filas_por_tramo(r, inc, horas, dict(base, detector=flujo, P_threshold=up,
                                                        S_threshold=us, detection_threshold=ud))
    return filas


def umbrales(r):
    if np.isfinite(r.detection_threshold):
        return f"D {r.detection_threshold:.2f}"
    return f"P {r.P_threshold:.2f} S {r.S_threshold:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["phasenet", "eqtransformer"],
                    choices=["phasenet", "eqtransformer"])
    args = ap.parse_args()

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zona = ix.unir(con_senal.ini.to_numpy(), con_senal.fin.to_numpy())
    z = np.load(lib.ruta_datos(FILTRO))
    valida = z["valida"]
    datos = {c: z[c] for c in ("Z", "N", "E")}
    horas = ix.horas_por_tramo(valida)

    filas, defectos = [], {}
    for modelo in args.modelos:
        m = an.cargar_modelo(modelo)
        dflt = por_defecto(m)
        defectos[modelo] = dict(dflt, blinding=dflt["blinding"][0])
        dentro, fuera = combinaciones(modelo, dflt)
        print(f"\n=== {modelo}: por defecto overlap {dflt['overlap']}, blinding {dflt['blinding']}, "
              f"stacking {dflt['stacking']} (ventana de {m.in_samples} muestras)")
        print(f"  {len(dentro)} combinaciones; se omiten {len(fuera)} con overlap < 2 x blinding "
              "(dejan tramos sin prediccion)")

        guardadas = dict(np.load(lib.ruta_anot(modelo, FILTRO)))
        t0 = time.time()
        for n, (o, b, s) in enumerate(dentro, 1):
            curvas = anotar(m, datos, o, b, s)
            es_def = (o, b, s) == (int(dflt["overlap"]), dflt["blinding"][0], dflt["stacking"])
            if es_def:
                dif = max(float(np.nanmax(np.abs(curvas[k] - guardadas[k]))) for k in guardadas)
                print(f"  comprobacion: la combinacion por defecto repite las curvas guardadas "
                      f"(diferencia maxima {dif:.2e}) -> {'SI' if dif < 1e-4 else 'NO, revisar'}")
            base = dict(modelo=modelo, overlap=o, blinding=b, stacking=s, por_defecto=es_def)
            for flujo in FLUJOS[modelo]:
                filas += barrer_umbrales(curvas, flujo, inc, zona, valida, horas, base)
            print(f"  {n}/{len(dentro)} overlap {o}, blinding {b}, stacking {s} "
                  f"({time.time() - t0:.0f} s)", flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(lib.DIR_SALIDA, "grilla_barrido.csv"), index=False)
    with open(os.path.join(lib.DIR_SALIDA, "grilla_por_defecto.json"), "w") as f:
        json.dump({k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                   for k, v in defectos.items()}, f, indent=2)

    todo = df[df.tramo.astype(str) == "todos"]
    print("\n1. POR COMBINACION DE VENTANAS (traza completa): maximo y menos FP/h con 0,95")
    print(f"   {'flujo':<25} {'overlap':>7} {'blind':>5} {'stack':>5} {'max':>6} {'FP/h':>6} "
          f"{'umbrales del max':>22} {'0,95: FP/h':>10} {'umbrales':>18}")
    for flujo, g in todo.groupby("detector", sort=False):
        for (o, b, s), h in g.groupby(["overlap", "blinding", "stacking"], sort=True):
            mx = ix.elegir(h, "max")
            ok = h[h.sens_volc >= 0.95]
            u_mx = umbrales(mx)
            if ok.empty:
                c95 = f"{'no':>10} {'':>18}"
            else:
                r = ok.sort_values(["fp_h", "sens_volc"], ascending=[True, False]).iloc[0]
                c95 = f"{r.fp_h:>10.1f} {umbrales(r):>18}"
            marca = "  <- por defecto" if h.por_defecto.iloc[0] else ""
            print(f"   {flujo:<25} {o:>7} {b:>5} {s:>5} {mx.sens_volc:>6.3f} {mx.fp_h:>6.1f} {u_mx:>22} {c95}{marca}")
        print()

    print("2. MEJOR DE TODA LA GRILLA (traza completa)")
    for flujo, g in todo.groupby("detector", sort=False):
        mx = ix.elegir(g, "max")
        d = g[g.por_defecto]
        mxd = ix.elegir(d, "max")
        print(f"   {flujo}: maximo {mx.k_volc}/{mx.n_volc} = {mx.sens_volc:.3f} con {ix.fmt_umbral(mx)}, "
              f"{mx.fp_h:.1f} FP/h (ventanas por defecto: {mxd.sens_volc:.3f}, {mxd.fp_h:.1f} FP/h)")
        for meta in ix.METAS:
            ok = g[g.sens_volc >= meta]
            okd = d[d.sens_volc >= meta]
            if ok.empty:
                print(f"     {meta:.2f}: no se alcanza")
                continue
            r = ok.sort_values(["fp_h", "sens_volc"], ascending=[True, False]).iloc[0]
            cmp = "no se alcanza" if okd.empty else f"{okd.fp_h.min():.1f} FP/h"
            print(f"     {meta:.2f}: {ix.fmt_umbral(r)} -> {r.k_volc}/{r.n_volc}, {r.fp_h:.1f} FP/h "
                  f"(VT {r.k_VT}, LP {r.k_LP}, TR {r.k_TR}); con ventanas por defecto: {cmp}")

    print("\n3. ROTACION DE TRAMOS SOBRE TODA LA GRILLA (se elige con dos tramos y se mide en el tercero)")
    print("   el tramo 3 tiene muy pocos volcanicos: su medicion pesa poco en la suma")
    dr = ix.rotar(df, CLAVES)
    dr.to_csv(os.path.join(lib.DIR_SALIDA, "grilla_rotacion.csv"), index=False)
    for (det, crit), g in dr.groupby(["detector", "criterio"], sort=False):
        print(f"   {det} [{crit}]")
        for r in g.itertuples():
            if r.tramo_medido == "suma":
                print(f"     suma: {r.k_medido}/{r.n_medido} = {r.k_medido / r.n_medido:.3f}, {r.fp_h_medido:.1f} FP/h")
            else:
                print(f"     mide tramo {r.tramo_medido}: {ix.fmt_umbral(r)} -> {r.k_medido}/{r.n_medido}, "
                      f"{r.fp_h_medido:.1f} FP/h")
    print(f"\nTablas en {lib.DIR_SALIDA}/grilla_barrido.csv, grilla_rotacion.csv y grilla_por_defecto.json")


if __name__ == "__main__":
    main()
