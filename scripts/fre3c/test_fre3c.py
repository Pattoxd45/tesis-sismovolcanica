#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para fre3c_lib.py y preparar_fre3c.py.

Uso: python test_fre3c.py
"""

import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fre3c_lib as lib

FS = lib.FS
S = int(FS)
RESULTADOS = []


def informar(nombre, ok, detalle=""):
    RESULTADOS.append(bool(ok))
    print(f"  [{'PASA ' if ok else 'FALLA'}] {nombre}" + (f"\n          {detalle}" if detalle else ""))


def campana(n, centro, ancho, alto):
    x = np.arange(n)
    return alto * np.exp(-0.5 * ((x - centro) / ancho) ** 2)


def test_disparos():
    print("\n1. Disparos")
    n = 20 * S
    # dos arribos a 5 s y 7 s, picos 0,9 y 0,6, valle entre ambos ~0,25
    c = campana(n, 5 * S, 40, 0.9) + campana(n, 7 * S, 40, 0.6)
    ini, fin, pico, valor = lib.disparos(c, 0.5)
    informar("umbral 0,5: dos disparos con picos en 5 s y 7 s",
             list(pico) == [5 * S, 7 * S], f"picos={list(pico)}")
    ini, fin, pico, valor = lib.disparos(c, 0.02)
    informar("umbral 0,02 (apagado 0,01): los dos arribos se funden en uno",
             list(pico) == [5 * S], f"picos={list(pico)}")
    informar("cache 0,02 filtrada a 0,5 conserva un solo arribo (pierde el de 7 s)",
             int((valor >= 0.5).sum()) == 1)
    c2 = c.copy()
    c2[:2 * S] = np.nan
    informar("NaN se trata como cero", len(lib.disparos(c2, 0.5)[0]) == 2)
    curva = np.zeros(30 * S)
    curva[:15 * S] = campana(15 * S, 14.9 * S, 30, 0.8)
    curva[15 * S:] = campana(15 * S, 0.1 * S, 30, 0.8)
    _, _, p, _ = lib.disparos_por_tramo(curva, 0.5, [(0, 15 * S), (15 * S, 30 * S)])
    informar("disparos_por_tramo no funde arribos a ambos lados del corte",
             len(p) == 2, f"picos={list(p)}")


def test_agrupar():
    print("\n2. Agrupacion anclada de P y S")
    g = lib.agrupar_anclado([1002 + 2 * S, 1000 + 4 * S + 2], 5 * S)
    informar("P a 2 s y S a 4 s: un grupo representado por la P",
             list(g) == [1002 + 2 * S], f"{list(g)}")
    t = np.arange(0, 21 * S, 4 * S)  # 0, 4, 8, 12, 16, 20 s
    g = lib.agrupar_anclado(t, 5 * S)
    informar("arribos cada 4 s durante 20 s: tres grupos, sin encadenar",
             list(g) == [0, 8 * S, 16 * S], f"{list(g)}")


def test_emparejar():
    print("\n3. Emparejamiento uno a uno")
    tol = 5 * S
    a, u = lib.emparejar([10_000], [10_000 + 490], tol)
    informar("deteccion a 4,9 s del inicio: acierta", a[0] == 0)
    a, u = lib.emparejar([10_000], [10_000 + 510], tol)
    informar("deteccion a 5,1 s del inicio: no acierta", a[0] == -1)
    a, u = lib.emparejar([10_000, 10_300], [10_150], tol)
    informar("dos eventos a 3 s y una deteccion entre ambos: un solo acierto",
             int((a >= 0).sum()) == 1 and a[0] == 0, f"asignado={list(a)}")
    a, u = lib.emparejar([10_000], [10_050, 10_200], tol)
    informar("dos detecciones para un evento: acierta la mas cercana",
             a[0] == 0 and list(u) == [True, False])


def test_evaluar():
    print("\n4. Clasificacion de detecciones y metricas")
    n = 2000 * S
    valida = np.ones(n, dtype=bool)
    valida[1500 * S:1600 * S] = False  # zona no valida
    eventos = pd.DataFrame([
        dict(ini=100 * S, fin=190 * S, clase="TR"),  # TR de 90 s
        dict(ini=300 * S, fin=315 * S, clase="VT"),
        dict(ini=500 * S, fin=508 * S, clase="IC"),
        dict(ini=700 * S, fin=730 * S, clase="LP"),  # no se detecta
    ])
    zona_ini = np.r_[eventos.ini.to_numpy() - 5 * S]
    zona_fin = np.r_[eventos.fin.to_numpy()]
    onsets = np.array([
        101 * S,
        130 * S,
        301 * S,
        303 * S,
        500 * S,
        900 * S,
        1000 * S,
        1550 * S,  # zona no valida: se descarta
    ])
    res = lib.evaluar(onsets, eventos, zona_ini, zona_fin, valida)
    cats = dict(zip(res["onsets"].tolist(), res["cat"].tolist()))
    esperado = {101 * S: "acierto", 130 * S: "interna", 301 * S: "acierto",
                303 * S: "interna", 500 * S: "acierto", 900 * S: "externa",
                1000 * S: "externa"}
    informar("categorias por deteccion", cats == esperado, f"{cats}")
    horas = valida.sum() / FS / 3600
    filas, tot = lib.resumir(res, eventos, horas)
    f = {r["clase"]: r for r in filas}
    informar("sensibilidad volcanica 2/3 (TR y VT si, LP no)",
             (f["volcanico"]["k"], f["volcanico"]["n"]) == (2, 3))
    informar("IC 1/1 cuenta en su clase y no en la volcanica",
             (f["IC"]["k"], f["IC"]["n"]) == (1, 1) and tot["aciertos_no_volc"] == 1)
    informar("FP = 2 externas; FDR = 2/(2+2) = 0,5; FDRest = 1 - 2/7",
             tot["externas"] == 2 and abs(tot["fdr"] - 0.5) < 1e-12
             and abs(tot["fdr_estricto"] - (1 - 2 / 7)) < 1e-12, f"{tot}")
    informar("FP/h usa solo horas validas",
             abs(tot["fp_h"] - 2 / horas) < 1e-9)
    informar("desfase del acierto TR = +1 s",
             abs(res["dt"][0] - 1.0) < 1e-12)
    me = eventos.clase.isin(["TR"]).to_numpy()
    mo = res["onsets"] < 200 * S
    filas, tot = lib.resumir(res, eventos, horas, me, mo)
    f = {r["clase"]: r for r in filas}
    informar("subconjunto: TR 1/1, 2 detecciones (1 acierto, 1 interna)",
             (f["TR"]["k"], f["TR"]["n"]) == (1, 1) and tot["detecciones"] == 2
             and tot["aciertos_volc"] == 1 and tot["internas"] == 1, f"{tot}")


def test_wilson_y_mascaras():
    print("\n5. Intervalos, mascaras y desplazamiento")
    lo, hi = lib.wilson(0, 10)
    informar("Wilson con k=0 empieza en 0", abs(lo) < 1e-12 and 0 < hi < 0.35)
    lo, hi = lib.wilson(37, 37)
    informar("Wilson con k=n termina en 1", abs(hi - 1) < 1e-12 and lo < 1)
    g = lib.mascara_guardas(10_000, [5_000], 1.0)
    informar("guarda de 1 s alrededor del corte", (~g).sum() == 200 and not g[4_950] and g[4_899])
    x = np.arange(5.0)
    informar("desplazar +1: y[k] = x[k-1]", list(lib.desplazar(x, 1, -1.0)) == [-1, 0, 1, 2, 3])
    informar("desplazar -1: y[k] = x[k+1]", list(lib.desplazar(x, -1, -1.0)) == [1, 2, 3, 4, -1])
    d = lib.dentro_de_intervalos([5, 15, 25, 35], [0, 10, 30], [12, 20, 30])
    informar("dentro_de_intervalos con intervalos solapados", list(d) == [True, True, False, False])


def test_preparar_tramo():
    print("\n6. Preparacion de un tramo sintetico (huecos y desfase de 1 muestra)")
    try:
        import preparar_fre3c as pf
    except Exception as ex:
        informar("importar preparar_fre3c", False, f"{type(ex).__name__}: {ex}")
        return
    from scipy.signal import sosfiltfilt
    rng = np.random.default_rng(1)
    n = 600 * S
    t0 = 1_500_000_000.0
    t_full = t0 + np.arange(n) / FS
    base = {c: rng.normal(0, 100, n) for c in "ZNE"}
    i_ev = 300 * S
    base["Z"][i_ev:i_ev + 200] += 3000 * np.sin(2 * np.pi * 5 * np.arange(200) / FS)
    # dos huecos conocidos en el crudo
    quitar = np.zeros(n, dtype=bool)
    quitar[100 * S:110 * S] = True
    quitar[450 * S:451 * S] = True
    crudo = {c: base[c][~quitar].astype(np.int32) for c in "ZNE"}
    crudo["tiempos_unix"] = {c: t_full[~quitar] for c in "ZNE"}
    crudo["fs"] = FS
    crudo["estacion"] = "FRE"
    # fila 1 simulada: Z filtrada y desplazada una muestra
    sos15 = lib.sos_filtro(lib.FILTROS["f1_15"])
    zf = sosfiltfilt(sos15, base["Z"].astype(np.int32).astype(float))
    fila1 = np.zeros(n)
    fila1[1:] = zf[:-1]
    zen = np.vstack([t_full, fila1])

    original = pf.cargar_crudo
    pf.cargar_crudo = lambda nombre: crudo
    try:
        out, valida, info = pf.preparar_tramo("sintetico", 1, 0, n, zen, sos15)
    finally:
        pf.cargar_crudo = original
    informar("desfase medido = +1 muestra", info["desfase_muestras"] == 1, f"{info['desfase_muestras']}")
    informar("huecos y su margen de 20 s quedan no validos",
             not valida[90 * S:130 * S].any() and not valida[440 * S:470 * S].any()
             and valida[200 * S:400 * S].all())
    z = sosfiltfilt(sos15, out["Z"])
    m = valida & (fila1 != 0)
    m[:20 * S] = False  # bordes del tramo
    m[-20 * S:] = False
    r = np.corrcoef(z[m], fila1[m])[0, 1]
    informar("tras corregir el desfase, Z filtrada reproduce la fila 1 lejos de bordes "
             "y huecos (r > 0,99999)", r > 0.99999, f"r={r:.8f}")
    k = int(np.argmax(np.abs(z[i_ev - 500:i_ev + 700]))) + i_ev - 500
    informar("el evento queda en la muestra de Zenodo (desplazado +1)",
             i_ev + 1 <= k <= i_ev + 201, f"maximo en {k}")


def main():
    print("TEST fre3c")
    try:
        import obspy
        test_disparos()
    except ImportError:
        print("\n1. Disparos: OMITIDO (falta obspy en este entorno)")
    test_agrupar()
    test_emparejar()
    test_evaluar()
    test_wilson_y_mascaras()
    test_preparar_tramo()
    n_ok = sum(RESULTADOS)
    print(f"\n{n_ok} de {len(RESULTADOS)} casos pasan")
    if n_ok != len(RESULTADOS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
