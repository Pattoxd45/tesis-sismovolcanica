#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fre3c_lib.py
============

Funciones comunes de la evaluacion de deteccion con UNA estacion, FRE, y
sus tres componentes reales (Z, N, E) tomadas de los datos crudos.

Lo usan:
    preparar_fre3c.py      alinea y filtra los crudos, marca eventos
    anotar_fre3c.py        corre PhaseNet y EQTransformer y guarda curvas
    sensibilidad_fre3c.py  tabla de sensibilidad por clase, FP/h y FDR
    test_fre3c.py          casos sinteticos de solucion conocida

DECISIONES QUE FIJA ESTE MODULO
-------------------------------
1. Disparos: se replican EXACTAMENTE los de SeisBench
   (WaveformModel.picks_from_annotations y detections_from_annotations):
   trigger_onset(curva, umbral, umbral / 2) de ObsPy, y el pico es el
   maximo de la curva entre el encendido y el apagado, ambos incluidos.
   Por eso se guardan las curvas completas y no los picks: extraer picks
   con umbral 0,02 y filtrarlos despues por confianza NO equivale a
   extraerlos con el umbral alto, porque con 0,02 el apagado ocurre recien
   bajo 0,01 y dos arribos cercanos quedan fundidos en un solo pick.
2. Emparejamiento: uno a uno, avido por |dt| creciente, con tolerancia
   de 5 s respecto del inicio catalogado. Es el mismo criterio que
   significancia.acertados_onset, para que las cifras sean comparables.
3. Union P y S: agrupacion ANCLADA. Un grupo empieza en el arribo mas
   temprano no asignado y absorbe los que caen a <= 5 s de ESE arribo,
   sin encadenar. El grupo se representa por su arribo mas temprano,
   porque el nivel 1 mide el inicio del evento.
4. Clasificacion de cada deteccion que no acierta:
     interna : cae en [inicio - 5 s, fin] de un evento catalogado con
               senal (incluido o no). Es senal real: no es falso positivo.
     externa : todo lo demas. Solo estas son falsos positivos.
   Los aciertos sobre AV o IC son aciertos de esa clase, no falsos
   positivos: son eventos reales que el clasificador debe mandar al
   rechazo.
5. FDR principal = externas / (aciertos volcanicos + externas).
   FDR estricto = 1 - aciertos volcanicos / detecciones (cota superior:
   cuenta como "no descubrimiento" todo lo que no es un acierto VT, LP
   o TR, incluidos los aciertos AV/IC y las internas).

Autor: Patricio Valdes Troncoso - Universidad Catolica de Temuco
"""

import os

import numpy as np

# =====================================================================
# CONSTANTES
# =====================================================================

FS = 100.0
N_TOTAL = 3_664_193

# Cortes entre tramos (los mismos de evaluar_detectores.py) y guarda
CORTES = [2_160_000, 3_240_001]
GUARDA_S = 30.0
LIMITES_TRAMO = [(0, 2_160_000), (2_160_000, 3_240_001),
                 (3_240_001, N_TOTAL)]

TOL_S = 5.0
CLASES = ["VT", "LP", "TR", "AV", "IC"]
VOLCANICAS = ["VT", "LP", "TR"]

# Filtros a evaluar: Butterworth orden 5, pasabanda, fase cero.
# f1_15 es el filtro de la traza de Zenodo; f1_45 es la banda con que se
# entreno EQTransformer.
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
    return butter(5, list(banda), btype="bandpass", fs=FS, output="sos")


# =====================================================================
# MASCARAS
# =====================================================================

def mascara_guardas(n=N_TOTAL, cortes=CORTES, guarda_s=GUARDA_S):
    """True fuera de las guardas de union (igual que evaluar_detectores)."""
    v = np.ones(n, dtype=bool)
    g = int(guarda_s * FS)
    for c in cortes:
        v[max(0, c - g):min(n, c + g)] = False
    return v


def desplazar(x, lag, relleno):
    """
    Devuelve y con y[k] = x[k - lag]. Convencion de verificar_crudos.py:
    desfase > 0 significa que la muestra n del crudo corresponde a la
    muestra n + desfase de Zenodo. Los extremos sin dato quedan en relleno.
    """
    y = np.full_like(x, relleno)
    if lag > 0:
        y[lag:] = x[:-lag]
    elif lag < 0:
        y[:lag] = x[-lag:]
    else:
        y[:] = x
    return y


# =====================================================================
# DISPAROS (replica de SeisBench)
# =====================================================================

def disparos(curva, umbral):
    """
    Replica de picks_from_annotations / detections_from_annotations de
    SeisBench sobre UNA curva contigua.

    Devuelve cuatro arreglos: inicio, fin, muestra del pico y valor del
    pico de cada disparo. Los NaN (sin anotacion) se tratan como 0.
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
    """Aplica disparos() por separado en cada tramo y devuelve indices globales."""
    partes = [[], [], [], []]
    for a, b in limites:
        ini, fin, pico, valor = disparos(curva[a:b], umbral)
        partes[0].append(ini + a)
        partes[1].append(fin + a)
        partes[2].append(pico + a)
        partes[3].append(valor)
    return tuple(np.concatenate(p) for p in partes)


def agrupar_anclado(t, tol):
    """
    Une arribos cercanos sin encadenar. t: muestras (cualquier orden).
    Un grupo empieza en el arribo mas temprano no asignado y absorbe los
    que estan a <= tol muestras de ese arribo. Devuelve el representante
    de cada grupo (su arribo mas temprano), ordenado.
    """
    t = np.sort(np.asarray(t, dtype=np.int64))
    out = []
    i = 0
    while i < t.size:
        out.append(t[i])
        i = int(np.searchsorted(t, t[i] + tol, side="right"))
    return np.asarray(out, dtype=np.int64)


# =====================================================================
# EMPAREJAMIENTO
# =====================================================================

def emparejar(inicios, onsets, tol):
    """
    Asignacion uno a uno, avida por |dt| creciente, dentro de +-tol
    muestras. Mismo criterio que significancia.acertados_onset (lista de
    candidatos (d, i, j) ordenada), pero buscando los candidatos con
    searchsorted para que sea rapido con muchos onsets.

    inicios: muestra de inicio de cada evento (orden del catalogo).
    onsets : muestras detectadas, ordenadas.
    Devuelve (indice del onset asignado a cada evento o -1,
              mascara de onsets asignados).
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
    """True para cada t que cae en algun [ini, fin] (intervalos cerrados)."""
    t = np.asarray(t, dtype=np.int64)
    if len(inis) == 0 or t.size == 0:
        return np.zeros(t.size, dtype=bool)
    orden = np.argsort(inis)
    inis = np.asarray(inis, dtype=np.int64)[orden]
    fins = np.asarray(fins, dtype=np.int64)[orden]
    # fusion de intervalos solapados
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
    """
    Empareja y clasifica.

    onsets   : muestras detectadas (se descartan las que caen en zona no
               valida, igual que coincidencia_onsets).
    eventos  : DataFrame de eventos INCLUIDOS con columnas ini, clase.
    zona_ini, zona_fin: intervalos de eventos con senal ([inicio - tol, fin])
               para decidir si una deteccion no asignada es interna.
    Devuelve dict con: onsets validos, vector de onset asignado por evento,
    dt por evento (s, NaN si no acierta) y categoria por onset
    ("acierto", "interna", "externa").
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
    """Intervalo de confianza del 95 % de Wilson para k aciertos de n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    medio = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(centro - medio), float(centro + medio)


def resumir(res, eventos, horas, mascara_eventos=None, mascara_onsets=None):
    """
    Resume una evaluacion, opcionalmente restringida a un subconjunto de
    eventos (por ejemplo, un tramo) y de onsets.

    Devuelve (filas por clase, fila de totales).
    """
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
    # aciertos volcanicos: onsets asignados a eventos VT, LP o TR del subconjunto
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
