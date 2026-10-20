#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Funciones comunes de la evaluacion de deteccion con FRE (Z, N, E reales)."""

import os

import numpy as np


FS = 100.0
N_TOTAL = 3_664_193

# mismos cortes y guarda que evaluar_detectores.py
CORTES = [2_160_000, 3_240_001]
GUARDA_S = 30.0
LIMITES_TRAMO = [(0, 2_160_000), (2_160_000, 3_240_001),
                 (3_240_001, N_TOTAL)]

TOL_S = 5.0
CLASES = ["VT", "LP", "TR", "AV", "IC"]
VOLCANICAS = ["VT", "LP", "TR"]

# 1-15 Hz: banda de la traza de Zenodo; 1-45 Hz: banda del filtro interno de EQTransformer
FILTROS = {"f1_15": (1.0, 15.0), "f1_45": (1.0, 45.0)}

UMBRALES = [0.02, 0.03, 0.05] + [round(0.05 * i, 2) for i in range(2, 20)]

DIR_TESIS = os.path.expanduser("~/tesis")
DIR_SALIDA = os.path.join(DIR_TESIS, "resultados", "fre3c")
NPY_ZENODO = os.path.join(DIR_TESIS, "datos", "zenodo",
                          "NVCh_10h_continuous_trace.npy")
CSV_CATALOGO = os.path.join(DIR_TESIS, "datos", "zenodo",
                            "NVCh_10h_continuous_trace_reference.csv")


def ruta_datos(filtro):
    return os.path.join(DIR_SALIDA, f"fre3c_{filtro}.npz")


def ruta_anot(modelo, filtro):
    return os.path.join(DIR_SALIDA, f"anot_{modelo}_{filtro}.npz")


RUTA_EVENTOS = os.path.join(DIR_SALIDA, "eventos_fre3c.csv")


def sos_filtro(banda):
    from scipy.signal import butter
    # orden 5 y fase cero: asi la Z cruda reproduce la fila 1 de Zenodo con r >= 0,99
    return butter(5, list(banda), btype="bandpass", fs=FS, output="sos")


def mascara_guardas(n=N_TOTAL, cortes=CORTES, guarda_s=GUARDA_S):
    v = np.ones(n, dtype=bool)
    g = int(guarda_s * FS)
    for c in cortes:
        v[max(0, c - g):min(n, c + g)] = False
    return v


def desplazar(x, lag, relleno):
    """y[k] = x[k - lag] (convencion de desfase de verificar_crudos.py)."""
    y = np.full_like(x, relleno)
    if lag > 0:
        y[lag:] = x[:-lag]
    elif lag < 0:
        y[:lag] = x[-lag:]
    else:
        y[:] = x
    return y


def disparos(curva, umbral):
    """Disparos como picks_from_annotations de SeisBench: encendido en umbral,
    apagado en umbral / 2, pico en el maximo. NaN cuenta como 0.
    """
    from obspy.signal.trigger import trigger_onset

    c = np.nan_to_num(np.asarray(curva, dtype=np.float64), nan=0.0)
    trig = trigger_onset(c, umbral, umbral / 2)
    if len(trig) == 0:
        vacio = np.array([], dtype=np.int64)
        return vacio, vacio, vacio, np.array([], dtype=float)
    trig = np.asarray(trig, dtype=np.int64)
    ini, fin = trig[:, 0], trig[:, 1]
    pico = np.empty(len(trig), dtype=np.int64)
    valor = np.empty(len(trig), dtype=float)
    for i, (s0, s1) in enumerate(trig):
        seg = c[s0:s1 + 1]
        k = int(np.argmax(seg))
        pico[i] = s0 + k
        valor[i] = float(seg[k])
    return ini, fin, pico, valor


def disparos_por_tramo(curva, umbral, limites=LIMITES_TRAMO):
    partes = [[], [], [], []]
    for a, b in limites:
        ini, fin, pico, valor = disparos(curva[a:b], umbral)
        partes[0].append(ini + a)
        partes[1].append(fin + a)
        partes[2].append(pico + a)
        partes[3].append(valor)
    return tuple(np.concatenate(p) for p in partes)


def agrupar_anclado(t, tol):
    """Une arribos a <= tol del primero del grupo, sin encadenar.
    Devuelve el arribo mas temprano de cada grupo.
    """
    t = np.sort(np.asarray(t, dtype=np.int64))
    out = []
    i = 0
    while i < t.size:
        out.append(t[i])
        i = int(np.searchsorted(t, t[i] + tol, side="right"))
    return np.asarray(out, dtype=np.int64)


def emparejar(inicios, onsets, tol):
    """Asignacion uno a uno por |dt| creciente dentro de +-tol (mismo criterio
    que significancia.acertados_onset).
    """
    inicios = np.asarray(inicios, dtype=np.int64)
    onsets = np.asarray(onsets, dtype=np.int64)
    cand = []
    for i, e in enumerate(inicios):
        a = int(np.searchsorted(onsets, e - tol, side="left"))
        b = int(np.searchsorted(onsets, e + tol, side="right"))
        for j in range(a, b):
            cand.append((abs(int(onsets[j]) - int(e)), i, j))
    cand.sort()
    asignado = np.full(inicios.size, -1, dtype=np.int64)
    usado = np.zeros(onsets.size, dtype=bool)
    for d, i, j in cand:
        if asignado[i] >= 0 or usado[j]:
            continue
        asignado[i] = j
        usado[j] = True
    return asignado, usado


def dentro_de_intervalos(t, inis, fins):
    t = np.asarray(t, dtype=np.int64)
    if len(inis) == 0 or t.size == 0:
        return np.zeros(t.size, dtype=bool)
    orden = np.argsort(inis)
    inis = np.asarray(inis, dtype=np.int64)[orden]
    fins = np.asarray(fins, dtype=np.int64)[orden]
    # se unen los intervalos solapados para que la busqueda con searchsorted sea correcta
    mi, mf = [inis[0]], [fins[0]]
    for a, b in zip(inis[1:], fins[1:]):
        if a <= mf[-1]:
            mf[-1] = max(mf[-1], b)
        else:
            mi.append(a)
            mf.append(b)
    mi, mf = np.asarray(mi), np.asarray(mf)
    k = np.searchsorted(mi, t, side="right") - 1
    ok = k >= 0
    res = np.zeros(t.size, dtype=bool)
    res[ok] = t[ok] <= mf[k[ok]]
    return res


def evaluar(onsets, eventos, zona_ini, zona_fin, valida, tol_s=TOL_S):
    """Empareja onsets con los inicios de los eventos incluidos y clasifica cada
    onset en acierto, interna (dentro de un evento con senal) o externa (FP).
    """
    tol = int(round(tol_s * FS))
    o = np.unique(np.asarray(onsets, dtype=np.int64))
    o = o[(o >= 0) & (o < valida.size)]
    o = o[valida[o]]
    inicios = eventos["ini"].to_numpy(dtype=np.int64)
    asignado, usado = emparejar(inicios, o, tol)
    dt = np.full(inicios.size, np.nan)
    ok = asignado >= 0
    dt[ok] = (o[asignado[ok]] - inicios[ok]) / FS
    cat = np.full(o.size, "externa", dtype=object)
    interna = dentro_de_intervalos(o, zona_ini, zona_fin)
    cat[interna] = "interna"
    cat[usado] = "acierto"
    return dict(onsets=o, asignado=asignado, dt=dt, cat=cat)


def wilson(k, n, z=1.959963984540054):
    """IC 95 % de Wilson."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    medio = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(centro - medio), float(centro + medio)


def resumir(res, eventos, horas, mascara_eventos=None, mascara_onsets=None):
    """Sensibilidad por clase y totales (FP/h, FDR) para un subconjunto."""
    if mascara_eventos is None:
        mascara_eventos = np.ones(len(eventos), dtype=bool)
    if mascara_onsets is None:
        mascara_onsets = np.ones(res["onsets"].size, dtype=bool)
    clases = eventos["clase"].to_numpy()
    acierta = res["asignado"] >= 0

    filas = []
    for c in CLASES + ["volcanico", "todas"]:
        if c == "volcanico":
            m = np.isin(clases, VOLCANICAS)
        elif c == "todas":
            m = np.ones(clases.size, dtype=bool)
        else:
            m = clases == c
        m &= mascara_eventos
        n = int(m.sum())
        k = int((acierta & m).sum())
        lo, hi = wilson(k, n)
        dts = res["dt"][m & acierta]
        filas.append(dict(clase=c, k=k, n=n,
                          sens=k / n if n else float("nan"),
                          ic_inf=lo, ic_sup=hi,
                          dt_mediana_s=float(np.median(dts)) if dts.size else float("nan")))

    cat = res["cat"][mascara_onsets]
    n_det = int(cat.size)
    volc_ev = np.isin(clases, VOLCANICAS) & mascara_eventos & acierta
    idx_volc = set(res["asignado"][volc_ev].tolist())
    idx_sel = np.where(mascara_onsets)[0]
    a_volc = sum(1 for j in idx_sel if j in idx_volc and res["cat"][j] == "acierto")
    a_total = int((cat == "acierto").sum())
    internas = int((cat == "interna").sum())
    externas = int((cat == "externa").sum())
    tot = dict(detecciones=n_det, aciertos_volc=a_volc,
               aciertos_no_volc=a_total - a_volc, internas=internas,
               externas=externas, horas=horas,
               fp_h=externas / horas if horas else float("nan"),
               fdr=externas / (a_volc + externas) if (a_volc + externas) else float("nan"),
               fdr_estricto=1 - a_volc / n_det if n_det else float("nan"))
    return filas, tot
