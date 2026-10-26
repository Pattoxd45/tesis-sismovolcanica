#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deteccion por interseccion: un evento cuenta si algun intervalo que entrega el
detector se cruza con [inicio, fin] del catalogo. Compara con +-5 s, barre los
umbrales de PhaseNet y EQTransformer y verifica la eleccion rotando los tramos.

Uso: python interseccion_fre3c.py [--filtro f1_15]
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
TOL = int(round(lib.TOL_S * FS))
UMBRAL_REF = 0.10
METAS = [0.80, 0.90, 0.95]
TRAMOS = [1, 2, 3]


def agrupar_intervalos(ini, fin, tol=TOL):
    """Agrupacion anclada de agrupar_anclado aplicada a intervalos: une los que
    empiezan a <= tol del primero del grupo; el grupo va del primer inicio al mayor fin.
    """
    orden = np.argsort(ini, kind="stable")
    ini, fin = np.asarray(ini, dtype=np.int64)[orden], np.asarray(fin, dtype=np.int64)[orden]
    gi, gf = [], []
    i = 0
    while i < ini.size:
        j = int(np.searchsorted(ini, ini[i] + tol, side="right"))
        gi.append(ini[i])
        gf.append(fin[i:j].max())
        i = j
    return np.asarray(gi, dtype=np.int64), np.asarray(gf, dtype=np.int64)


def unir(ini, fin):
    """Une intervalos solapados (solo para saber si un intervalo toca algun evento)."""
    if len(ini) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    o = np.argsort(ini)
    ini, fin = np.asarray(ini, dtype=np.int64)[o], np.asarray(fin, dtype=np.int64)[o]
    mi, mf = [ini[0]], [fin[0]]
    for a, b in zip(ini[1:], fin[1:]):
        if a <= mf[-1]:
            mf[-1] = max(mf[-1], b)
        else:
            mi.append(a)
            mf.append(b)
    return np.asarray(mi), np.asarray(mf)


def toca_zona(pi, pf, zi, zf):
    k = np.searchsorted(zi, pf, side="right") - 1
    ok = k >= 0
    res = np.zeros(pi.size, dtype=bool)
    res[ok] = zf[k[ok]] >= pi[ok]
    return res


def evaluar_interseccion(pi, pf, inc, zona, valida):
    """Asignacion uno a uno que maximiza los eventos detectados; cada intervalo
    cuenta para un solo evento. Intervalos cerrados: tocarse en una muestra ya cruza.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    pi, pf = np.asarray(pi, dtype=np.int64), np.asarray(pf, dtype=np.int64)
    m = (pi >= 0) & (pi < valida.size)
    m[m] = valida[pi[m]]
    pi, pf = pi[m], pf[m]
    o = np.argsort(pi, kind="stable")
    pi, pf = pi[o], pf[o]

    ei = inc["ini"].to_numpy(dtype=np.int64)
    ef = inc["fin"].to_numpy(dtype=np.int64)
    largo = int((pf - pi).max()) if pi.size else 0
    filas, cols = [], []
    for i, (a, b) in enumerate(zip(ei, ef)):
        lo = int(np.searchsorted(pi, a - largo, side="left"))
        hi = int(np.searchsorted(pi, b, side="right"))
        j = np.arange(lo, hi)
        j = j[pf[j] >= a]
        filas += [i] * j.size
        cols += j.tolist()

    asignado = np.full(ei.size, -1, dtype=np.int64)
    if filas:
        g = csr_matrix((np.ones(len(filas)), (filas, cols)), shape=(ei.size, pi.size))
        asignado = maximum_bipartite_matching(g, perm_type="column").astype(np.int64)
    usado = np.zeros(pi.size, dtype=bool)
    usado[asignado[asignado >= 0]] = True

    dt = np.full(ei.size, np.nan)
    ok = asignado >= 0
    dt[ok] = (pi[asignado[ok]] - ei[ok]) / FS
    cat = np.full(pi.size, "externa", dtype=object)
    cat[toca_zona(pi, pf, *zona)] = "interna"
    cat[usado] = "acierto"
    return dict(onsets=pi, fines=pf, asignado=asignado, dt=dt, cat=cat)


def intervalos_fase(curvas, etiqueta, umbral, cache):
    clave = (etiqueta, umbral)
    if clave not in cache:
        ini, fin, _, _ = lib.disparos_por_tramo(curvas[etiqueta], umbral)
        cache[clave] = (ini, fin)
    return cache[clave]


def intervalos_ps(curvas, up, us, cache):
    pi, pf = intervalos_fase(curvas, "P", up, cache)
    si, sf = intervalos_fase(curvas, "S", us, cache)
    return agrupar_intervalos(np.concatenate([pi, si]), np.concatenate([pf, sf]))


def intervalos_stalta(filtro):
    import evaluar_detectores as ed
    import recorte_fre3c as rc
    from obspy.signal.trigger import trigger_onset

    # sin la union a 5 s de la Etapa 2: el intervalo tal como lo entrega trigger_onset
    z = np.load(lib.ruta_datos(filtro))["Z"]
    ini, fin = [], []
    for (a, _), c in zip(lib.LIMITES_TRAMO, rc.cft_stalta(z)):
        for i, j in trigger_onset(c, ed.STALTA["thr_on"], ed.STALTA["thr_off"]):
            ini.append(a + int(i))
            fin.append(a + int(j))
    return np.asarray(ini, dtype=np.int64), np.asarray(fin, dtype=np.int64)


def tramo_de(t):
    return np.searchsorted(lib.CORTES, t, side="right") + 1


def horas_por_tramo(valida):
    h = {t: valida[a:b].sum() / FS / 3600 for t, (a, b) in zip(TRAMOS, lib.LIMITES_TRAMO)}
    h["todos"] = valida.sum() / FS / 3600
    return h


def filas_por_tramo(res, inc, horas, base):
    """Una fila por tramo y otra con la traza completa: aciertos por clase y FP."""
    tr_ev = inc["tramo"].to_numpy()
    tr_pr = tramo_de(res["onsets"])
    out = []
    for t in TRAMOS + ["todos"]:
        me = np.ones(len(inc), dtype=bool) if t == "todos" else tr_ev == t
        mp = np.ones(tr_pr.size, dtype=bool) if t == "todos" else tr_pr == t
        fc, ft = lib.resumir(res, inc, horas[t], me, mp)
        c = {f["clase"]: f for f in fc}
        out.append(dict(base, tramo=t, horas=horas[t],
                        **{f"k_{k}": c[k]["k"] for k in lib.VOLCANICAS},
                        **{f"n_{k}": c[k]["n"] for k in lib.VOLCANICAS},
                        k_volc=c["volcanico"]["k"], n_volc=c["volcanico"]["n"],
                        sens_volc=c["volcanico"]["sens"], fp=ft["externas"],
                        fp_h=ft["fp_h"], fdr=ft["fdr"]))
    return out


def comparar_criterios(inc, zona, con_senal, valida, filtro):
    import tolerancia_fre3c as tf

    pn = dict(np.load(lib.ruta_anot("phasenet", filtro)))
    eqt = dict(np.load(lib.ruta_anot("eqtransformer", filtro)))
    ivs = {"PhaseNet P y S": intervalos_ps(pn, UMBRAL_REF, UMBRAL_REF, {}),
           "EQTransformer deteccion": tuple(lib.disparos_por_tramo(eqt["Detection"], UMBRAL_REF)[:2]),
           "EQTransformer P y S": intervalos_ps(eqt, UMBRAL_REF, UMBRAL_REF, {}),
           "STA/LTA": intervalos_stalta(filtro)}
    ons = tf.onsets_detectores(UMBRAL_REF, filtro, valida)
    horas = horas_por_tramo(valida)
    filas, diag = [], []
    for det, (pi, pf) in ivs.items():
        r = evaluar_interseccion(pi, pf, inc, zona, valida)
        filas += [dict(f, criterio="interseccion") for f in filas_por_tramo(r, inc, horas, dict(detector=det))
                  if f["tramo"] == "todos"]
        for t in (5, 10):
            b = lib.evaluar(ons[det], inc, con_senal.ini.to_numpy() - t * int(FS),
                            con_senal.fin.to_numpy(), valida, t)
            filas += [dict(f, criterio=f"+-{t} s") for f in filas_por_tramo(b, inc, horas, dict(detector=det))
                      if f["tramo"] == "todos"]
            if t == 5:
                diag += diferencias(det, r, b, inc, pi, pf, ons[det])
        dur = (pf - pi) / FS
        print(f"  {det}: {pi.size} intervalos, duracion mediana {np.median(dur):.1f} s, p90 {np.percentile(dur, 90):.1f} s")
    return pd.DataFrame(filas), pd.DataFrame(diag)


def diferencias(det, r_int, r_tol, inc, pi, pf, onsets):
    """Volcanicos que un criterio detecta y el otro no. Si lo detecta la interseccion, el intervalo
    asignado; si lo detecta +-5 s, el aviso asignado y el intervalo mas cercano al inicio.
    """
    vol = inc.clase.isin(lib.VOLCANICAS).to_numpy()
    a_int, a_tol = r_int["asignado"] >= 0, r_tol["asignado"] >= 0
    oi, of, ot = r_int["onsets"], r_int["fines"], r_tol["onsets"]
    out = []
    for i in np.where(vol & (a_int != a_tol))[0]:
        e = inc.iloc[i]
        if a_int[i]:
            k = int(r_int["asignado"][i])
            j = int(np.argmin(np.abs(ot - e.ini))) if ot.size else None
        else:
            k = int(np.argmin(np.minimum(np.abs(oi - e.ini), np.abs(of - e.ini)))) if oi.size else None
            j = int(r_tol["asignado"][i])
        out.append(dict(detector=det, id=int(e.id), clase=e.clase, dur_s=(e.fin - e.ini) / FS,
                        solo="interseccion" if a_int[i] else "+-5 s",
                        intervalo_ini_rel_s=(oi[k] - e.ini) / FS if k is not None else np.nan,
                        intervalo_fin_rel_s=(of[k] - e.ini) / FS if k is not None else np.nan,
                        aviso_rel_s=(ot[j] - e.ini) / FS if j is not None else np.nan))
    return out

def barrer(inc, zona, valida, filtro):
    horas = horas_por_tramo(valida)
    filas = []
    for modelo, flujo in (("phasenet", "PhaseNet P y S"), ("eqtransformer", "EQTransformer P y S"),
                          ("eqtransformer", "EQTransformer deteccion")):
        curvas = dict(np.load(lib.ruta_anot(modelo, filtro)))
        cache = {}
        if flujo.endswith("deteccion"):
            combos = [(np.nan, np.nan, u) for u in lib.UMBRALES]
        else:
            combos = [(up, us, np.nan) for up in lib.UMBRALES for us in lib.UMBRALES]
        print(f"  {flujo}: {len(combos)} combinaciones", flush=True)
        for up, us, ud in combos:
            if flujo.endswith("deteccion"):
                pi, pf = intervalos_fase(curvas, "Detection", ud, cache)
            else:
                pi, pf = intervalos_ps(curvas, up, us, cache)
            r = evaluar_interseccion(pi, pf, inc, zona, valida)
            filas += filas_por_tramo(r, inc, horas, dict(detector=flujo, P_threshold=up,
                                                         S_threshold=us, detection_threshold=ud))
    return pd.DataFrame(filas)


def elegir(g, criterio):
    """g: una fila por combinacion. 'max': mayor sensibilidad, luego menos FP/h.
    'meta': menos FP/h que llega a 0,95; si ninguna llega, como 'max'.
    """
    if criterio == "meta" and (g.sens_volc >= 0.95).any():
        g = g[g.sens_volc >= 0.95]
        return g.sort_values(["fp_h", "sens_volc"], ascending=[True, False]).iloc[0]
    return g.sort_values(["sens_volc", "fp_h"], ascending=[False, True]).iloc[0]


CLAVES_UMBRAL = ["P_threshold", "S_threshold", "detection_threshold"]


def rotar(df, claves=CLAVES_UMBRAL):
    """Elige con dos tramos y mide en el tercero; suma las tres mediciones."""
    d0 = df[df.tramo.astype(str) != "todos"].copy()
    d0["tramo"] = d0.tramo.astype(int)
    d0["combo"] = ["|".join(t) for t in zip(*[d0[k].map(str) for k in claves])]
    filas = []
    for det, d in d0.groupby("detector"):
        for criterio in ("max", "meta"):
            k_tot = n_tot = fp_tot = 0
            horas_tot = 0.0
            for t in TRAMOS:
                ent = d[d.tramo != t].groupby("combo", as_index=False)[["k_volc", "n_volc", "fp", "horas"]].sum()
                ent["sens_volc"] = ent.k_volc / ent.n_volc
                ent["fp_h"] = ent.fp / ent.horas
                e = elegir(ent, criterio)
                p = d[(d.tramo == t) & (d.combo == e.combo)].iloc[0]
                k_tot += p.k_volc
                n_tot += p.n_volc
                fp_tot += p.fp
                horas_tot += p.horas
                filas.append(dict(detector=det, criterio=criterio, tramo_medido=t,
                                  **{k: p[k] for k in claves},
                                  sens_eleccion=e.sens_volc, fp_h_eleccion=e.fp_h,
                                  k_medido=int(p.k_volc), n_medido=int(p.n_volc), fp_h_medido=p.fp_h))
            filas.append(dict(detector=det, criterio=criterio, tramo_medido="suma",
                              k_medido=int(k_tot), n_medido=int(n_tot), fp_h_medido=fp_tot / horas_tot))
    return pd.DataFrame(filas)


def fmt_umbral(r):
    if np.isfinite(r.detection_threshold):
        txt = f"detection_threshold {r.detection_threshold:.2f}"
    else:
        txt = f"P {r.P_threshold:.2f}, S {r.S_threshold:.2f}"
    if "overlap" in getattr(r, "_fields", getattr(r, "index", [])):
        txt += f", overlap {int(r.overlap)}, blinding {int(r.blinding)}, stacking {r.stacking}"
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtro", default="f1_15", choices=list(lib.FILTROS))
    args = ap.parse_args()

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zona = unir(con_senal.ini.to_numpy(), con_senal.fin.to_numpy())
    valida = np.load(lib.ruta_datos("f1_15"))["valida"]

    print("EVENTOS INCLUIDOS POR TRAMO (AV e IC se informan, no entran en la sensibilidad)")
    print(pd.crosstab(inc.tramo, inc.clase, margins=True).to_string())
    for k, (a, b) in enumerate(lib.LIMITES_TRAMO, 1):
        print(f"  tramo {k}: {valida[a:b].sum() / FS / 3600:.2f} h con registro")

    print(f"\n1. INTERSECCION CONTRA TOLERANCIA (umbral {UMBRAL_REF:.2f} en las redes; STA/LTA de la Etapa 2)")
    dc, dd = comparar_criterios(inc, zona, con_senal, valida, args.filtro)
    print(f"   {'detector':<25} {'criterio':<13} {'VT':>6} {'LP':>6} {'TR':>6} {'volc':>8} {'sens':>6} {'FP/h':>6}")
    for r in dc.itertuples():
        print(f"   {r.detector:<25} {r.criterio:<13} {r.k_VT:>3}/{r.n_VT:<2} {r.k_LP:>3}/{r.n_LP:<2} "
              f"{r.k_TR:>3}/{r.n_TR:<2} {r.k_volc:>4}/{r.n_volc:<3} {r.sens_volc:>6.3f} {r.fp_h:>6.1f}")
    dc.to_csv(os.path.join(lib.DIR_SALIDA, "interseccion_criterios.csv"), index=False)

    print("\n   Volcanicos que detecta un criterio y no el otro (interseccion contra +-5 s), en s desde el inicio")
    print("   (intervalo: el asignado si detecta la interseccion, el mas cercano si no; aviso: el asignado si detecta +-5 s, el mas cercano si no)")
    print(f"   {'detector':<25} {'id':>4} {'clase':>5} {'dur':>6} {'solo con':>12} {'intervalo':>16} {'aviso':>7}")
    for r in dd.itertuples():
        iv = f"{r.intervalo_ini_rel_s:.1f} a {r.intervalo_fin_rel_s:.1f}" if np.isfinite(r.intervalo_ini_rel_s) else "-"
        print(f"   {r.detector:<25} {r.id:>4} {r.clase:>5} {r.dur_s:>6.1f} {r.solo:>12} {iv:>16} {r.aviso_rel_s:>7.1f}")
    dd.to_csv(os.path.join(lib.DIR_SALIDA, "interseccion_diferencias.csv"), index=False)

    print("\n2. BARRIDO DE UMBRALES CON INTERSECCION (traza completa)")
    df = barrer(inc, zona, valida, args.filtro)
    df.to_csv(os.path.join(lib.DIR_SALIDA, "interseccion_barrido.csv"), index=False)
    todo = df[df.tramo == "todos"]
    for det, g in todo.groupby("detector", sort=False):
        mx = elegir(g, "max")
        print(f"   {det}: maximo {mx.k_volc}/{mx.n_volc} = {mx.sens_volc:.3f} con {fmt_umbral(mx)}, {mx.fp_h:.1f} FP/h")
        for meta in METAS:
            ok = g[g.sens_volc >= meta]
            if ok.empty:
                print(f"     {meta:.2f}: no se alcanza")
                continue
            r = ok.sort_values(["fp_h", "sens_volc"], ascending=[True, False]).iloc[0]
            print(f"     {meta:.2f}: {fmt_umbral(r)} -> {r.k_volc}/{r.n_volc} = {r.sens_volc:.3f}, "
                  f"{r.fp_h:.1f} FP/h (VT {r.k_VT}, LP {r.k_LP}, TR {r.k_TR})")

    print("\n3. ROTACION DE TRAMOS: se elige con dos tramos y se mide en el tercero")
    print("   max = mayor sensibilidad y luego menos FP/h; meta = menos FP/h con 0,95 (si no, max)")
    dr = rotar(df)
    dr.to_csv(os.path.join(lib.DIR_SALIDA, "interseccion_rotacion.csv"), index=False)
    for (det, crit), g in dr.groupby(["detector", "criterio"], sort=False):
        print(f"   {det} [{crit}]")
        for r in g.itertuples():
            if r.tramo_medido == "suma":
                print(f"     suma de los tres tramos medidos: {r.k_medido}/{r.n_medido} = "
                      f"{r.k_medido / r.n_medido:.3f}, {r.fp_h_medido:.1f} FP/h")
            else:
                print(f"     mide tramo {r.tramo_medido}: elegido {fmt_umbral(r)} "
                      f"(en los otros dos {r.sens_eleccion:.3f}, {r.fp_h_eleccion:.1f} FP/h) -> "
                      f"{r.k_medido}/{r.n_medido}, {r.fp_h_medido:.1f} FP/h")
    print(f"\nTablas en {lib.DIR_SALIDA}/interseccion_criterios.csv, interseccion_diferencias.csv, "
          "interseccion_barrido.csv e interseccion_rotacion.csv")


if __name__ == "__main__":
    main()
