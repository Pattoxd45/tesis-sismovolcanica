#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostico_tramos.py
=====================

Diagnostico por tramo de la traza continua. Solo MIDE: no ajusta ningun
parametro ni modifica ningun script. Toda la configuracion (canales,
cortes, STA/LTA, tolerancias, IoU minimo, ventana y paso del barrido) se
toma tal cual de evaluacion/evaluar_detectores.py y de ventanas/barrido.py.

Que imprime:

  1. Por tramo: cuantos eventos tiene el catalogo en cada clase y la
     sensibilidad de nivel 2 de STA/LTA por clase sobre la traza completa.
     Se lee de resultados/ventanas/comparacion_barrido.json, que ya
     contiene ese desglose (lo escribe ventanas/ventanas_traza_real.py).

  2. Por tramo y por canal: fraccion de muestras exactamente iguales a
     cero y desviacion estandar. Sirve para saber si un tramo con poca
     sensibilidad tiene senal utilizable en los cinco canales o no.

  3. Las detecciones de red que difieren entre la corrida sobre la traza
     completa y la corrida con barrido de 5 minutos: inicio y fin en
     segundos desde el inicio de su tramo, duracion, distancia al borde
     de ventana mas cercano y evento del catalogo con el que se solapa
     (clase e IoU). Esto exige recalcular las dos corridas, porque el
     JSON guarda metricas agregadas y no la lista de detecciones.

Uso:
    python diagnostico_tramos.py
    python diagnostico_tramos.py --sin-cache     # recalcula la parte 3

La parte 3 tarda varios minutos (STA/LTA sobre 10 h, dos veces). El
resultado se guarda en resultados/ventanas/detecciones_red.json y se
reutiliza en las corridas siguientes salvo que se pida --sin-cache.

Autor: Patricio Valdes Troncoso - Universidad Catolica de Temuco
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "ventanas"))
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
import barrido as br                   # noqa: E402
import evaluar_detectores as ed        # noqa: E402
import ventanas_traza_real as vtr      # noqa: E402

FS = ed.FS

JSON_COMPARACION = os.path.join(ed.SALIDA, "ventanas", "comparacion_barrido.json")
JSON_DETECCIONES = os.path.join(ed.SALIDA, "ventanas", "detecciones_red.json")


# =====================================================================
# UTILIDADES
# =====================================================================

def utc(t):
    return datetime.fromtimestamp(float(t), timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S")


def tramo_de_muestra(m, segmentos):
    """Numero de tramo (base 1) y muestra inicial del tramo."""
    for k, (t0, t1) in enumerate(segmentos, start=1):
        if t0 <= m < t1:
            return k, t0
    raise ValueError("muestra fuera de la traza")


def distancia_al_borde(m, bordes):
    """Distancia en muestras al borde de ventana mas cercano."""
    j = int(np.searchsorted(bordes, m))
    cand = []
    if j > 0:
        cand.append(abs(m - int(bordes[j - 1])))
    if j < len(bordes):
        cand.append(abs(int(bordes[j]) - m))
    return min(cand)


def evento_solapado(d, catalogo):
    """Evento del catalogo con mayor IoU entre los que se solapan con d."""
    sol = [(ed.iou(e, d), e) for e in catalogo
           if e.ini < d.fin and d.ini < e.fin]
    if not sol:
        return None, 0.0
    v, e = max(sol, key=lambda x: x[0])
    return e, v


# =====================================================================
# 1. CATALOGO Y SENSIBILIDAD POR TRAMO Y CLASE
# =====================================================================

def parte1(res):
    """Lee el desglose por tramo y clase del JSON de comparacion."""
    print("=" * 78)
    print("  1. CATALOGO Y SENSIBILIDAD DE NIVEL 2 POR TRAMO Y CLASE")
    print(f"     STA/LTA sobre la traza completa, IoU >= {ed.IOU_MIN:g}")
    print(f"     Fuente: {os.path.relpath(JSON_COMPARACION, '/home/patto/tesis')}")
    print("=" * 78)

    tramos = res["completa"]["por_tramo"]

    print("\n  Eventos del catalogo")
    print("    tramo  inicio UTC                 h" +
          "".join(f"{c:>6}" for c in ed.CLASES) + f"{'total':>8}")
    for r in tramos:
        pc = r["por_clase"]
        horas = r["muestras"] / FS / 3600.0
        print(f"    {r['tramo']:>5}  {r['inicio_utc']}  {horas:>6.2f}" +
              "".join(f"{pc[c]['n'] if c in pc else 0:>6}" for c in ed.CLASES) +
              f"{r['n_catalogo']:>8}")
    g = res["completa"]["global"]["por_clase"]
    print(f"    {'todos':>5}  {'':<19}  {'':>6}" +
          "".join(f"{g[c]['n'] if c in g else 0:>6}" for c in ed.CLASES) +
          f"{res['completa']['global']['n_catalogo']:>8}")

    print("\n  Sensibilidad de nivel 2 (recall; '-' = sin eventos de esa clase)")
    print("    tramo" + "".join(f"{c:>8}" for c in ed.CLASES) + f"{'global':>9}")
    for r in tramos:
        pc = r["por_clase"]
        print(f"    {r['tramo']:>5}" +
              "".join(f"{pc[c]['recall']:>8.3f}" if c in pc else f"{'-':>8}"
                      for c in ed.CLASES) +
              f"{r['recall']:>9.3f}")
    print(f"    {'todos':>5}" +
          "".join(f"{g[c]['recall']:>8.3f}" if c in g else f"{'-':>8}"
                  for c in ed.CLASES) +
          f"{res['completa']['global']['recall']:>9.3f}")

    print("\n  Eventos del catalogo no recuperados, por tramo y clase")
    print("    tramo" + "".join(f"{c:>8}" for c in ed.CLASES) + f"{'total':>9}")
    for r in tramos:
        pc = r["por_clase"]
        faltan = {c: round(pc[c]["n"] * (1.0 - pc[c]["recall"]))
                  for c in ed.CLASES if c in pc}
        print(f"    {r['tramo']:>5}" +
              "".join(f"{faltan[c]:>8}" if c in faltan else f"{'-':>8}"
                      for c in ed.CLASES) +
              f"{sum(faltan.values()):>9}")


# =====================================================================
# 2. CALIDAD DE SENAL POR TRAMO Y CANAL
# =====================================================================

def parte2(a, segmentos):
    """Fraccion de muestras exactamente cero y desviacion estandar."""
    print("\n" + "=" * 78)
    print("  2. SENAL POR TRAMO Y CANAL: CEROS EXACTOS Y DESVIACION ESTANDAR")
    print("     Medido sobre el tramo entero (las guardas de union son 60 s")
    print("     de 1,2 a 6 h, no mueven estas cifras)")
    print("=" * 78)
    print(f"\n    tramo  canal     muestras   frac. cero          sigma")
    filas = []
    for k, (t0, t1) in enumerate(segmentos, start=1):
        for ch in ed.CANALES:
            x = np.asarray(a[ch, t0:t1], dtype=np.float64)
            frac = float(np.mean(x == 0.0))
            sigma = float(np.std(x))
            filas.append({"tramo": k, "canal": ch, "muestras": int(t1 - t0),
                          "frac_cero": frac, "sigma": sigma})
            print(f"    {k:>5}  {ch:>5}  {t1 - t0:>11}  {frac:>10.6f}"
                  f"  {sigma:>13.4f}")
        print()

    print("    Razon de sigma entre tramos, por canal (tramo / tramo 1)")
    print("    canal" + "".join(f"{'tramo ' + str(k):>12}"
                                for k in range(1, len(segmentos) + 1)))
    for ch in ed.CANALES:
        s = {f["tramo"]: f["sigma"] for f in filas if f["canal"] == ch}
        print(f"    {ch:>5}" + "".join(
            f"{s[k] / s[1]:>12.3f}" if s.get(1) else f"{'-':>12}"
            for k in range(1, len(segmentos) + 1)))
    return filas


# =====================================================================
# 3. DETECCIONES QUE DIFIEREN ENTRE COMPLETA Y BARRIDO
# =====================================================================

def calcular_detecciones(a, valida, ventanas):
    """Detecciones de red de las dos corridas, con la logica ya existente."""
    print("  STA/LTA sobre la traza completa ...")
    det_c = ed.coincidencia(ed.detector_stalta(a, ed.CANALES), valida)
    print("  STA/LTA con barrido de 5 minutos ...")
    pc_b, _ = vtr.stalta_barrido(a, ed.CANALES, ventanas)
    det_b = ed.coincidencia(pc_b, valida)
    return det_c, det_b


def obtener_detecciones(a, valida, ventanas, usar_cache):
    """Lee las detecciones del cache o las recalcula y las guarda."""
    if usar_cache and os.path.exists(JSON_DETECCIONES):
        with open(JSON_DETECCIONES, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("config", {}).get("stalta") == ed.STALTA:
            print(f"  Detecciones leidas de {JSON_DETECCIONES}")
            return ([ed.Intervalo(i, j) for i, j in d["completa"]],
                    [ed.Intervalo(i, j) for i, j in d["barrido"]])
        print("  El cache no corresponde a la configuracion actual; recalculo")
    det_c, det_b = calcular_detecciones(a, valida, ventanas)
    os.makedirs(os.path.dirname(JSON_DETECCIONES), exist_ok=True)
    with open(JSON_DETECCIONES, "w", encoding="utf-8") as f:
        json.dump({"config": {"stalta": ed.STALTA, "canales": ed.CANALES,
                              "ventana_s": br.VENTANA_S, "paso_s": br.PASO_S},
                   "completa": [[d.ini, d.fin] for d in det_c],
                   "barrido": [[d.ini, d.fin] for d in det_b]},
                  f, indent=2, default=float)
    return det_c, det_b


def parte3(a, catalogo, det_c, det_b, ventanas, segmentos, res):
    print("\n" + "=" * 78)
    print("  3. DETECCIONES DE RED QUE DIFIEREN ENTRE COMPLETA Y BARRIDO")
    print("=" * 78)

    # Control: las cuentas recalculadas deben coincidir con el JSON.
    for nombre, det in (("completa", det_c), ("barrido", det_b)):
        esperado = res[nombre]["global"]["n_detecciones"]
        marca = "ok" if len(det) == esperado else "DISCREPA"
        print(f"\n  Control [{marca}]: {nombre} tiene {len(det)} detecciones; "
              f"el JSON informa {esperado}")

    bordes = np.array(sorted({b for v in ventanas for b in v}))
    print(f"\n  Rejilla de bordes de ventana: {len(bordes)} bordes distintos, "
          f"separacion mediana {np.median(np.diff(bordes)) / FS:.1f} s")

    sc = {(d.ini, d.fin) for d in det_c}
    sb = {(d.ini, d.fin) for d in det_b}
    solo_c = sorted(sc - sb)
    solo_b = sorted(sb - sc)
    print(f"  Identicas en las dos corridas: {len(sc & sb)}")
    print(f"  Solo en la traza completa    : {len(solo_c)}")
    print(f"  Solo en el barrido           : {len(solo_b)}")

    for origen, pares, otros in (("solo en la traza completa", solo_c, det_b),
                                 ("solo en el barrido", solo_b, det_c)):
        print(f"\n  {'-' * 74}")
        print(f"  Detecciones {origen}")
        print(f"  {'-' * 74}")
        if not pares:
            print("    ninguna")
            continue
        print(f"    {'tr':>3}{'ini_s':>11}{'fin_s':>11}{'dur_s':>9}"
              f"{'d.borde_ini':>13}{'d.borde_fin':>13}  catalogo")
        for ini, fin in pares:
            d = ed.Intervalo(ini, fin)
            k, t0 = tramo_de_muestra(ini, segmentos)
            ev, v = evento_solapado(d, catalogo)
            if ev is None:
                txt = "sin solape con el catalogo"
            else:
                marca = ">=" if v >= ed.IOU_MIN else "< "
                txt = (f"{ev.clase} de {ev.dur / FS:.1f} s, IoU {v:.3f} "
                       f"({marca} {ed.IOU_MIN:g})")
            print(f"    {k:>3}{(ini - t0) / FS:>11.2f}{(fin - t0) / FS:>11.2f}"
                  f"{d.dur / FS:>9.2f}"
                  f"{distancia_al_borde(ini, bordes) / FS:>13.2f}"
                  f"{distancia_al_borde(fin, bordes) / FS:>13.2f}  {txt}")
            # Contraparte en la otra corrida: distingue una deteccion nueva
            # de un corrimiento de los limites de una ya existente.
            cont = [(ed.iou(d, o), o) for o in otros
                    if o.ini < d.fin and d.ini < o.fin]
            if cont:
                v2, o = max(cont, key=lambda x: x[0])
                print(f"        contraparte en la otra corrida: "
                      f"{(o.ini - t0) / FS:.2f} a {(o.fin - t0) / FS:.2f} s "
                      f"({o.dur / FS:.2f} s), IoU {v2:.3f}")
            else:
                print("        contraparte en la otra corrida: ninguna")


# =====================================================================
# MAIN
# =====================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=JSON_COMPARACION)
    ap.add_argument("--sin-cache", action="store_true",
                    help="recalcula las detecciones en vez de leer el cache")
    args = ap.parse_args(argv)

    if not os.path.exists(args.json):
        sys.exit(f"Falta {args.json}. Corre primero "
                 f"scripts/ventanas/ventanas_traza_real.py")
    with open(args.json, encoding="utf-8") as f:
        res = json.load(f)

    print("Cargando traza y catalogo ...")
    a, catalogo, valida, horas = ed.cargar()
    n = a.shape[-1]
    segmentos = ed.segmentos(n)
    ventanas = br.generar_ventanas(n, ed.CORTES)
    print(f"  {len(catalogo)} eventos, {horas:.2f} h validas, "
          f"{len(ventanas)} ventanas, canales {ed.CANALES}\n")

    parte1(res)
    parte2(a, segmentos)

    print("\n  Calculando las detecciones de red de las dos corridas ...")
    det_c, det_b = obtener_detecciones(a, valida, ventanas, not args.sin_cache)
    parte3(a, catalogo, det_c, det_b, ventanas, segmentos, res)


if __name__ == "__main__":
    main()
