#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barrido de P_threshold y S_threshold por separado, igual para PhaseNet y EQTransformer,
y eventos que se detectan solo con P, solo con S o con ambas.

Uso: python barrido_ps_fre3c.py [--filtros f1_15 f1_45]
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
MODELOS = ("phasenet", "eqtransformer")
POR_DEFECTO = {"phasenet": 0.3, "eqtransformer": 0.1}  # valor de SeisBench para P y S
UMBRALES_TABLA = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
UMBRALES_DIF = [0.05, 0.10, 0.30]
METAS = [0.80, 0.90, 0.95]
NOMBRE_FILTRO = {"f1_15": "1-15 Hz", "f1_45": "1-45 Hz"}
FLUJO_PS = {"phasenet": "PN-PS", "eqtransformer": "EQT-PS"}


def picos(curvas, fase, umbral, cache):
    clave = (fase, umbral)
    if clave not in cache:
        cache[clave] = lib.disparos_por_tramo(curvas[fase], umbral)[2]
    return cache[clave]


def barrer(modelo, filtro, curvas, inc, zona_ini, zona_fin, valida, horas):
    cache, filas = {}, []
    for up in lib.UMBRALES:
        for us in lib.UMBRALES:
            # P y S a 5 s o menos del primero cuentan como un aviso, igual que en PN-PS
            o = lib.agrupar_anclado(np.concatenate([picos(curvas, "P", up, cache),
                                                    picos(curvas, "S", us, cache)]), TOL)
            res = lib.evaluar(o, inc, zona_ini, zona_fin, valida)
            fc, ft = lib.resumir(res, inc, horas)
            c = {f["clase"]: f for f in fc}
            filas.append(dict(modelo=modelo, filtro=filtro, P_threshold=up, S_threshold=us,
                              **{f"k_{k}": c[k]["k"] for k in lib.CLASES},
                              sens_volc=c["volcanico"]["sens"], ic_inf=c["volcanico"]["ic_inf"],
                              ic_sup=c["volcanico"]["ic_sup"], fp_h=ft["fp_h"], fdr=ft["fdr"]))
    return pd.DataFrame(filas), cache


def aciertos_fase(curvas, fase, umbral, cache, inc, zona_ini, zona_fin, valida):
    res = lib.evaluar(picos(curvas, fase, umbral, cache), inc, zona_ini, zona_fin, valida)
    return res["asignado"] >= 0


def matriz(df, col, fmt):
    print(f"   {'P/S':>6} " + " ".join(f"{u:>6.2f}" for u in UMBRALES_TABLA))
    for up in UMBRALES_TABLA:
        fila = df[np.isclose(df.P_threshold, up)]
        celdas = [fmt(fila[np.isclose(fila.S_threshold, us)][col].iloc[0]) for us in UMBRALES_TABLA]
        print(f"   {up:>6.2f} " + " ".join(f"{x:>6}" for x in celdas))


def mejor(df, meta):
    ok = df[df.sens_volc >= meta]
    if ok.empty:
        return None
    return ok.sort_values(["fp_h", "P_threshold", "S_threshold"],
                          ascending=[True, False, False]).iloc[0]


def texto_fila(r):
    if r is None:
        return "no se alcanza"
    return (f"P {r.P_threshold:.2f} / S {r.S_threshold:.2f}: {r.sens_volc:.3f} "
            f"[{r.ic_inf:.3f}; {r.ic_sup:.3f}], {r.fp_h:.1f} FP/h, FDR {r.fdr:.3f}")


def verificar_diagonal(df, modelo, filtro):
    ruta_c = os.path.join(lib.DIR_SALIDA, "sens_clase.csv")
    ruta_t = os.path.join(lib.DIR_SALIDA, "sens_totales.csv")
    if not (os.path.exists(ruta_c) and os.path.exists(ruta_t)):
        return "sin sens_clase.csv: no se puede comparar"
    sc, st = pd.read_csv(ruta_c), pd.read_csv(ruta_t)
    fl = FLUJO_PS[modelo]
    c = sc[(sc.flujo == fl) & (sc.filtro == filtro) & (sc.tramo.astype(str) == "todos")
           & (sc.clase == "volcanico")]
    t = st[(st.flujo == fl) & (st.filtro == filtro) & (st.tramo.astype(str) == "todos")]
    d = df[np.isclose(df.P_threshold, df.S_threshold)]
    malos = 0
    for r in d.itertuples():
        a = c[np.isclose(c.umbral, r.P_threshold)]
        b = t[np.isclose(t.umbral, r.P_threshold)]
        k = r.k_VT + r.k_LP + r.k_TR
        if a.empty or b.empty or a.k.iloc[0] != k \
                or not np.isclose(b.fp_h.iloc[0], r.fp_h):
            malos += 1
    return f"{len(d) - malos}/{len(d)} umbrales iguales a {fl} de sensibilidad_fre3c.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtros", nargs="+", default=list(lib.FILTROS), choices=list(lib.FILTROS))
    args = ap.parse_args()

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    valida = np.load(lib.ruta_datos("f1_15"))["valida"]
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zona_ini, zona_fin = (con_senal.ini - TOL).to_numpy(), con_senal.fin.to_numpy()
    horas = valida.sum() / FS / 3600
    volc = inc.clase.isin(lib.VOLCANICAS).to_numpy()

    tablas, por_evento = [], inc[["id", "clase", "tramo", "dur_s"]].copy()
    print("BARRIDO DE P_threshold Y S_threshold POR SEPARADO (nivel 1, tolerancia 5 s)")
    print("  sensibilidad volcanica = VT+LP+TR detectados / 125; FP/h fuera de todo evento;")
    print("  FDR = FP / (aciertos volcanicos + FP). Filas: P_threshold; columnas: S_threshold.")
    for modelo in MODELOS:
        for filtro in args.filtros:
            z = np.load(lib.ruta_anot(modelo, filtro))
            curvas = {k: np.nan_to_num(z[k]) for k in ("P", "S")}
            df, cache = barrer(modelo, filtro, curvas, inc, zona_ini, zona_fin, valida, horas)
            tablas.append(df)
            print(f"\n=== {modelo}, filtro {NOMBRE_FILTRO[filtro]} ===")
            print(f"  comprobacion de la diagonal: {verificar_diagonal(df, modelo, filtro)}")
            print("\n  Sensibilidad volcanica:")
            matriz(df, "sens_volc", lambda v: f"{v:.3f}")
            print("\n  Falsas alarmas por hora:")
            matriz(df, "fp_h", lambda v: f"{v:.1f}")

            d = POR_DEFECTO[modelo]
            r = df[np.isclose(df.P_threshold, d) & np.isclose(df.S_threshold, d)].iloc[0]
            print(f"\n  Por defecto de SeisBench: {texto_fila(r)}")
            diag = df[np.isclose(df.P_threshold, df.S_threshold)]
            mx = df.sort_values(["sens_volc", "fp_h"], ascending=[False, True]).iloc[0]
            mxd = diag.sort_values(["sens_volc", "fp_h"], ascending=[False, True]).iloc[0]
            print(f"  Maximo con el mismo umbral para P y S: {texto_fila(mxd)}")
            print(f"  Maximo con umbrales separados:         {texto_fila(mx)}")
            print("  Menos FP/h que alcanza cada meta:")
            for meta in METAS:
                print(f"    {meta:.2f}  mismo umbral: {texto_fila(mejor(diag, meta))}")
                print(f"          separados:    {texto_fila(mejor(df, meta))}")

            print("\n  Eventos volcanicos segun la fase que los detecta (P y S con el mismo umbral):")
            print(f"   {'umbral':>6} {'clase':>5} {'ambas':>6} {'solo P':>7} {'solo S':>7} {'ninguna':>8}")
            for u in UMBRALES_DIF:
                hp = aciertos_fase(curvas, "P", u, cache, inc, zona_ini, zona_fin, valida)
                hs = aciertos_fase(curvas, "S", u, cache, inc, zona_ini, zona_fin, valida)
                por_evento[f"{modelo}_{filtro}_P_{u:.2f}"] = hp
                por_evento[f"{modelo}_{filtro}_S_{u:.2f}"] = hs
                for cl in lib.VOLCANICAS:
                    m = (inc.clase == cl).to_numpy()
                    print(f"   {u:>6.2f} {cl:>5} {int((hp & hs & m).sum()):>6} "
                          f"{int((hp & ~hs & m).sum()):>7} {int((~hp & hs & m).sum()):>7} "
                          f"{int((~hp & ~hs & m).sum()):>8}")
                if np.isclose(u, 0.10):
                    sp = inc[hp & ~hs & volc]
                    ss = inc[~hp & hs & volc]
                    print(f"          solo P: {', '.join(f'{x.id}:{x.clase}' for x in sp.itertuples()) or '-'}")
                    print(f"          solo S: {', '.join(f'{x.id}:{x.clase}' for x in ss.itertuples()) or '-'}")

    pd.concat(tablas).to_csv(os.path.join(lib.DIR_SALIDA, "barrido_ps.csv"), index=False)
    por_evento.to_csv(os.path.join(lib.DIR_SALIDA, "ps_por_evento.csv"), index=False)
    print(f"\nTablas en {lib.DIR_SALIDA}/barrido_ps.csv y ps_por_evento.csv")


if __name__ == "__main__":
    main()
