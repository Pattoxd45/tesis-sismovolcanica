#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barrido.py
==========

Barrido con ventanas de 5 minutos, fusion de intervalos y recorte
adaptativo. Implementa el diseno de la seccion 2.4 del informe (tarea
B-08).

    1. Barrido : ventanas de 300 s con paso de 55 s, sin cruzar los cortes
                 entre tramos de la traza. Como 55 <= 300 - 240,8, todo
                 evento del catalogo cabe entero en al menos una ventana.
    2. Deteccion: el detector recibe cada ventana por separado y sus
                 intervalos se trasladan a indices globales.
    3. Fusion  : se unen los intervalos que se solapan o que estan
                 separados por menos de un umbral. Asi una deteccion
                 cortada en el borde de una ventana se completa con su
                 continuacion en la siguiente.
    4. Recorte : cada intervalo fusionado se recorta con duracion adaptada
                 y un margen previo que tolera el error de onset medido.

Todo trabaja sobre indices de muestra, nunca sobre tiempo Unix. Las
funciones del nucleo son puras (no leen archivos) y se prueban con casos
sinteticos en test_barrido.py.

Autor: Patricio Valdes Troncoso - Universidad Catolica de Temuco
"""

import numpy as np

FS = 100.0            # Hz

VENTANA_S = 300.0     # ventana de barrido
PASO_S = 55.0         # paso entre ventanas
DUR_MAX_CAT_S = 240.8 # evento mas largo del catalogo
GAP_FUSION_S = 5.0    # igual a FUSION_S de evaluar_detectores.py
MARGEN_PRE_S = 2.5    # cubre el error de onset medido (1,7 a 2,4 s)
MARGEN_POST_S = 0.0   # el diseno solo pide margen previo
VENTANA_FIJA_S = 81.92  # 8.192 muestras, ventana del pool de entrenamiento

# La condicion de diseno se verifica al importar: si alguien cambia los
# parametros y la rompe, el modulo falla de inmediato.
assert PASO_S <= VENTANA_S - DUR_MAX_CAT_S, "paso incompatible con la ventana"


# =====================================================================
# 1. BARRIDO
# =====================================================================

def ventanas_de_tramo(ini, fin, ventana, paso):
    """
    Ventanas [a, a + ventana) dentro del tramo [ini, fin).

    Si el tramo no termina justo en una ventana regular, se agrega una
    ultima ventana alineada al final del tramo para que la cola quede
    cubierta con la misma garantia. Un tramo mas corto que la ventana
    produce una sola ventana, el tramo completo.
    """
    if fin - ini <= ventana:
        return [(ini, fin)]
    inicios = list(range(ini, fin - ventana + 1, paso))
    if inicios[-1] + ventana < fin:
        inicios.append(fin - ventana)
    return [(a, a + ventana) for a in inicios]


def generar_ventanas(n, cortes, ventana_s=VENTANA_S, paso_s=PASO_S):
    """Ventanas de toda la traza, tramo por tramo, sin cruzar cortes."""
    v, p = int(round(ventana_s * FS)), int(round(paso_s * FS))
    bordes = [0] + list(cortes) + [n]
    salida = []
    for i in range(len(bordes) - 1):
        salida.extend(ventanas_de_tramo(bordes[i], bordes[i + 1], v, p))
    return salida


def ventanas_que_contienen(ini, fin, ventanas):
    """Indices de las ventanas que contienen entero el intervalo."""
    return [k for k, (a, b) in enumerate(ventanas) if a <= ini and fin <= b]


def ventanas_que_tocan(ini, fin, ventanas):
    """Indices de las ventanas que se solapan con el intervalo."""
    return [k for k, (a, b) in enumerate(ventanas) if a < fin and ini < b]


# =====================================================================
# 2. DETECCION POR VENTANA
# =====================================================================

def detectar_por_ventanas(senal, ventanas, detector):
    """
    Aplica `detector` a cada ventana y devuelve los intervalos en indices
    globales.

    senal    : arreglo 1D de la traza completa de un canal (puede ser un
               memmap; solo se lee cada ventana).
    detector : funcion x -> lista de (ini, fin) relativos a la ventana.

    Cada intervalo se devuelve como dict con la ventana de origen y si toca
    un borde. Una deteccion que toca un borde puede estar cortada; la fusion
    la completa con la misma deteccion vista desde otra ventana.
    """
    salida = []
    for k, (a, b) in enumerate(ventanas):
        x = np.asarray(senal[a:b], dtype=np.float64)
        for s, e in detector(x):
            salida.append({"ini": a + int(s), "fin": a + int(e), "ventana": k,
                           "toca_borde_izq": int(s) <= 0,
                           "toca_borde_der": int(e) >= (b - a) - 1})
    return salida


# =====================================================================
# 3. FUSION
# =====================================================================

def fusionar(intervalos, gap):
    """
    Une intervalos que se solapan o estan separados por menos de `gap`
    muestras. Recibe tuplas (ini, fin) y devuelve tuplas ordenadas.
    """
    if not intervalos:
        return []
    orden = sorted((int(s), int(e)) for s, e in intervalos)
    out = [list(orden[0])]
    for s, e in orden[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


# =====================================================================
# 4. RECORTE ADAPTATIVO
# =====================================================================

def tramo_de(muestra, n, cortes):
    """Limites [ini, fin) del tramo que contiene la muestra."""
    bordes = [0] + list(cortes) + [n]
    for i in range(len(bordes) - 1):
        if bordes[i] <= muestra < bordes[i + 1]:
            return bordes[i], bordes[i + 1]
    raise ValueError("muestra fuera de la traza")


def recorte_adaptativo(ini, fin, n, cortes, margen_pre_s=MARGEN_PRE_S,
                       margen_post_s=MARGEN_POST_S):
    """
    Recorte con la duracion del intervalo mas los margenes, sin salir del
    tramo al que pertenece (un recorte nunca mezcla dos fechas).
    """
    t0, t1 = tramo_de(ini, n, cortes)
    a = max(t0, ini - int(round(margen_pre_s * FS)))
    b = min(t1, fin + int(round(margen_post_s * FS)))
    return a, b


def recorte_fijo(ini, n, cortes, margen_pre_s=MARGEN_PRE_S,
                 dur_s=VENTANA_FIJA_S):
    """Ventana fija de 81,92 s desde el onset, para comparar."""
    t0, t1 = tramo_de(ini, n, cortes)
    a = max(t0, ini - int(round(margen_pre_s * FS)))
    return a, min(t1, a + int(round(dur_s * FS)))


# =====================================================================
# DETECTOR STA/LTA PARA UNA VENTANA
# =====================================================================

def stalta_en_ventana(x, cfg):
    """
    STA/LTA clasico sobre una ventana, con la misma configuracion que
    evaluar_detectores.py. Intervalos relativos a la ventana.

    La LTA necesita lta_s de senal antes de dar una razon valida, asi que
    los primeros lta_s segundos de cada ventana no disparan. Con paso de
    55 s y LTA de 60 s, esos segundos ya estan cubiertos por la ventana
    anterior, donde la LTA si esta calentada.
    """
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from obspy.signal.filter import bandpass

    if np.allclose(x, 0) or len(x) < int(cfg["lta_s"] * FS) * 2:
        return []
    xf = bandpass(x, cfg["banda"][0], cfg["banda"][1], df=FS,
                  corners=4, zerophase=True)
    cft = classic_sta_lta(xf, int(cfg["sta_s"] * FS), int(cfg["lta_s"] * FS))
    cft = np.nan_to_num(cft, nan=0.0, posinf=0.0, neginf=0.0)
    return [(int(s), int(e)) for s, e in
            trigger_onset(cft, cfg["thr_on"], cfg["thr_off"])]
