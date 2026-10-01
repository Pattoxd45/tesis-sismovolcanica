#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snr_eventos.py
==============

Razon senal/ruido de los eventos del catalogo, cruzada con si cada
detector los encuentra. Responde la pregunta que Ricardo dejo planteada
en la reunion del 21 de septiembre: cuando un detector pierde un evento,
es que el detector falla o es que el evento esta enterrado en ruido.

Solo MIDE. No ajusta ningun parametro, no modifica ningun script y no
propone ningun cambio de configuracion.

DEFINICION DE LA RAZON
----------------------
Se reutiliza tal cual la de revisar_falsos_positivos.py: pico absoluto
del evento sobre el RMS de los 60 s inmediatamente anteriores, en
decibeles, sobre la traza SIN filtrar.

    razon_dB = 20 * log10( max|x_evento| / rms(x_ruido) )

Esa definicion toma la ventana de ruido completa, sin mirar que hay
dentro. Si en esos 60 s hay otro evento del catalogo, el RMS del "ruido"
sube y la razon del evento sale artificialmente baja. Si hay muestras en
cero (huecos de dato), el RMS baja y la razon sale artificialmente alta.
Por eso se agrega una segunda variante:

    base      la de revisar_falsos_positivos.py, sin tocar.
    estricta  misma formula, pero antes de calcular el RMS se sacan de
              la ventana de ruido las muestras que caen dentro de otro
              evento del catalogo y las muestras exactamente iguales a
              cero. Si quedan menos de 5 s limpios, el evento queda sin
              razon en vez de reportar un numero contaminado.

El script reporta cuantos eventos cambian entre una y otra, y cuanto.

VALIDACION ANTES DE USAR DATOS REALES
-------------------------------------
La formula se prueba primero contra un caso de solucion conocida: una
senoide de amplitud A sobre ruido gaussiano de sigma declarado, donde el
resultado tiene que aproximarse a 20*log10(A/sigma). El script se detiene
si no pasa. La prueba tambien mide dos sesgos del estimador: el que
aparece cuando A y sigma son comparables, y el PISO que impone la
duracion del evento, porque el pico de un tramo largo de puro ruido es
mayor que el de uno corto. Los dos importan para leer la tabla por
rangos.

RAZON DE RED
------------
Se calcula una razon por cada canal utilizable que tenga dato para ese
evento, y la razon de red es la MEDIANA entre canales. La mediana y no el
promedio porque una estacion con ganancia distinta o con un hueco parcial
arrastraria el promedio.

Uso:
    python snr_eventos.py
    python snr_eventos.py --sin-cache      # recalcula STA/LTA
    python snr_eventos.py --solo-autotest

Autor: Patricio Valdes Troncoso - Universidad Catolica de Temuco
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
import evaluar_detectores as ed            # noqa: E402
import significancia as sig                # noqa: E402
import evaluar_por_tramo as ept            # noqa: E402
import revisar_falsos_positivos as rfp     # noqa: E402

FS = ed.FS
SALIDA = ed.SALIDA

# Ventana de ruido y minimo de ruido limpio exigido, los dos tomados de
# revisar_falsos_positivos.snr_intervalo para no cambiar la definicion.
MARGEN_RUIDO_S = 60.0
MIN_RUIDO_S = 5.0

# Duraciones a las que se mide el piso del estimador, en segundos. Van
# del evento mas corto del catalogo (3,4 s) al mas largo (240,8 s), e
# incluyen la ventana fija del pool de entrenamiento (81,92 s).
DURACIONES_PISO = [3.4, 10.0, 30.0, 81.92, 150.0, 240.8]

# Rangos de razon para la tabla de sensibilidad, en dB.
RANGOS = [(-np.inf, 5.0, "< 5"),
          (5.0, 10.0, "5 a 10"),
          (10.0, 20.0, "10 a 20"),
          (20.0, np.inf, "> 20")]

DETECTORES_N1 = ["stalta", "phasenet", "eqtransformer"]
DETECTORES_N2 = ["stalta", "eqtransformer"]

CSV_SALIDA = os.path.join(SALIDA, "snr_eventos.csv")

# Diferencia a partir de la cual se considera que un evento "cambia" de
# una variante a la otra. 0,1 dB esta muy por debajo de la dispersion
# entre canales y no se confunde con ruido numerico.
TOL_CAMBIO_DB = 0.1


# =====================================================================
# VALIDACION CON CASO SINTETICO DE SOLUCION CONOCIDA
# =====================================================================

def autotest(verbose=True):
    """
    Senoide de amplitud A sobre ruido gaussiano de sigma conocida.

    Con la definicion pico sobre RMS, el valor exacto es:
        pico de la senoide            = A
        RMS del ruido gaussiano       -> sigma
        razon                         -> 20*log10(A/sigma)

    Dos casos:
      1. Evento SIN ruido superpuesto. El unico error posible es que el
         RMS muestral del ruido no sea exactamente sigma, y con 60 s a
         100 Hz ese error es de milesimas de dB. Se exige 0,15 dB.
      2. Evento CON ruido superpuesto, que es el caso real. Aqui el pico
         medido es max|A*sen + ruido| >= A, asi que el estimador sobre-
         estima. El sesgo no se exige: se MIDE y se muestra, porque es
         justo lo que afecta al rango de razon baja.
    """
    rng = np.random.default_rng(20260101)
    n_ruido = int(MARGEN_RUIDO_S * FS)
    n_ev = int(20.0 * FS)
    t = np.arange(n_ev) / FS
    sigma = 3.0
    ok = True
    piso = {}

    if verbose:
        print("=" * 78)
        print("  VALIDACION CON CASO SINTETICO DE SOLUCION CONOCIDA")
        print("=" * 78)
        print(f"  Senoide de 5 Hz y amplitud A, {n_ev / FS:g} s, sobre ruido")
        print(f"  gaussiano de sigma = {sigma:g}; ventana de ruido de "
              f"{MARGEN_RUIDO_S:g} s.")
        print(f"\n  Caso 1: el evento NO lleva ruido superpuesto")
        print(f"  {'A':>10}{'A/sigma':>10}{'esperado':>11}{'medido':>10}"
              f"{'error':>10}")

    for A in [3.0, 9.0, 30.0, 300.0, 3000.0]:
        ruido = sigma * rng.standard_normal(n_ruido)
        x_ev = A * np.sin(2 * np.pi * 5.0 * t)
        esperado = 20.0 * np.log10(A / sigma)
        medido = rfp.snr(x_ev, ruido)
        err = medido - esperado
        if abs(err) > 0.15:
            ok = False
        if verbose:
            print(f"  {A:>10.1f}{A / sigma:>10.1f}{esperado:>11.2f}"
                  f"{medido:>10.2f}{err:>+10.3f}"
                  + ("" if abs(err) <= 0.15 else "   FALLA"))

    if verbose:
        print(f"\n  Caso 2: el evento lleva el mismo ruido superpuesto")
        print("  (el pico medido es max|A*sen + ruido| >= A, asi que el")
        print("  estimador sobreestima; el sesgo se mide, no se exige)")
        print(f"  {'A':>10}{'A/sigma':>10}{'esperado':>11}{'medido':>10}"
              f"{'sesgo':>10}")
        for A in [3.0, 6.0, 9.0, 30.0, 300.0, 3000.0]:
            ruido = sigma * rng.standard_normal(n_ruido)
            x_ev = (A * np.sin(2 * np.pi * 5.0 * t)
                    + sigma * rng.standard_normal(n_ev))
            esperado = 20.0 * np.log10(A / sigma)
            medido = rfp.snr(x_ev, ruido)
            print(f"  {A:>10.1f}{A / sigma:>10.1f}{esperado:>11.2f}"
                  f"{medido:>10.2f}{medido - esperado:>+10.3f}")

        print("\n  Caso 3: PISO del estimador con el evento de puro ruido")
        print("  Si el 'evento' es solo ruido, la razon verdadera es 0 dB,")
        print("  pero el pico es el maximo de la gaussiana en la ventana y")
        print("  crece con la duracion. Ese piso es el minimo que puede")
        print("  reportar el estimador para un evento de esa duracion.")
        print("  Las duraciones son las del catalogo real: 3,4 s el evento")
        print("  mas corto y 240,8 s el mas largo.")
        print(f"  {'dur_s':>10}{'muestras':>10}{'piso_dB':>10}"
              f"{'p10':>8}{'p90':>8}   (100 repeticiones)")
        for dur_s in DURACIONES_PISO:
            n_d = int(dur_s * FS)
            vals = []
            for _ in range(100):
                ruido = sigma * rng.standard_normal(n_ruido)
                x_ev = sigma * rng.standard_normal(n_d)
                vals.append(rfp.snr(x_ev, ruido))
            vals = np.asarray(vals)
            piso[dur_s] = float(np.median(vals))
            print(f"  {dur_s:>10.2f}{n_d:>10}{np.median(vals):>10.2f}"
                  f"{np.percentile(vals, 10):>8.2f}"
                  f"{np.percentile(vals, 90):>8.2f}")

        print("\n  Caso 4: control de casos degenerados")
        print(f"    ruido todo cero       -> {rfp.snr(np.ones(10), np.zeros(10))}"
              "  (nan, correcto)")
        print(f"    evento todo cero      -> {rfp.snr(np.zeros(10), np.ones(10))}"
              "  (nan, correcto)")
        if not np.isnan(rfp.snr(np.ones(10), np.zeros(10))):
            ok = False
        if not np.isnan(rfp.snr(np.zeros(10), np.ones(10))):
            ok = False

        print(f"\n  Caso 1 dentro de 0,15 dB: {'SI' if ok else 'NO'}")
    return ok, piso


# =====================================================================
# RAZON POR EVENTO
# =====================================================================

def mascara_eventos(n):
    """
    Booleano por muestra: True si la muestra cae dentro de un evento.

    Se construye con el catalogo COMPLETO del CSV (205 eventos), no con
    los 203 que quedan tras excluir las guardas de union. Los dos
    excluidos siguen siendo senal sismica real, y si caen dentro de la
    ventana de ruido de un evento vecino la contaminan igual. Para
    limpiar el ruido importa donde hay eventos, no cuales se evaluan.
    """
    df = pd.read_csv(ed.CSV)
    m = np.zeros(n, dtype=bool)
    n_ev = 0
    for r in df.itertuples():
        i0, i1 = int(r.idx_start), int(r.idx_end)
        m[max(0, i0):min(n, i1)] = True
        n_ev += 1
    return m, n_ev


def snr_evento(a, canal, ini, fin, estricta, ocupado=None):
    """
    Razon de un evento en un canal.

    estricta=False  reproduce revisar_falsos_positivos.snr_intervalo.
    estricta=True   saca de la ventana de ruido las muestras que caen
                    dentro de OTRO evento del catalogo (mascara ocupado)
                    y las muestras exactamente cero, y exige que queden
                    al menos MIN_RUIDO_S segundos limpios.
    """
    if not estricta:
        return rfp.snr_intervalo(a, canal, ini, fin,
                                 margen_s=MARGEN_RUIDO_S)

    m = int(MARGEN_RUIDO_S * FS)
    f0 = max(0, ini - m)
    if ini - f0 < int(MIN_RUIDO_S * FS):
        return float("nan")
    x_ev = np.asarray(a[canal, ini:fin], dtype=np.float64)
    x_bg = np.asarray(a[canal, f0:ini], dtype=np.float64)
    if x_ev.size == 0 or x_bg.size == 0:
        return float("nan")

    limpio = x_bg != 0.0
    if ocupado is not None:
        # ocupado marca TODO evento; la ventana es anterior al evento en
        # estudio, asi que lo que marque ahi es otro evento del catalogo.
        limpio &= ~ocupado[f0:ini]
    if limpio.sum() < int(MIN_RUIDO_S * FS):
        return float("nan")
    return rfp.snr(x_ev, x_bg[limpio])


def razones(a, catalogo, canales, ocupado):
    """
    Razon por canal y razon de red (mediana entre canales) de cada
    evento, en las dos variantes.
    """
    filas = []
    for k, ev in enumerate(catalogo):
        fila = {"evento": k}
        for etiqueta, estricta in (("base", False), ("est", True)):
            vals = {}
            for ch in canales:
                # "canal con datos": si el tramo del evento es todo cero
                # en ese canal, el canal no tiene dato ahi.
                x = np.asarray(a[ch, ev.ini:ev.fin], dtype=np.float64)
                if x.size == 0 or np.allclose(x, 0):
                    vals[ch] = float("nan")
                    continue
                vals[ch] = snr_evento(a, ch, ev.ini, ev.fin, estricta,
                                      ocupado)
            fin = [v for v in vals.values() if np.isfinite(v)]
            for ch in canales:
                fila[f"snr_{etiqueta}_ch{ch}"] = vals[ch]
            fila[f"n_canales_{etiqueta}"] = len(fin)
            fila[f"snr_{etiqueta}_db"] = (float(np.median(fin)) if fin
                                          else float("nan"))
        filas.append(fila)
    return pd.DataFrame(filas)


# =====================================================================
# DETECCIONES EN LOS PUNTOS DE OPERACION DE significancia.py
# =====================================================================

def banderas_deteccion(a, catalogo, valida, horas, canales, usar_cache):
    """
    Vector booleano por evento, por detector y por nivel.

    Reutiliza los puntos de operacion y la reconstruccion de salidas de
    significancia.py a traves de evaluar_por_tramo.globales, y verifica
    contra los valores citados antes de devolver nada.
    """
    print("\n" + "=" * 78)
    print("  DETECCIONES EN LOS PUNTOS DE OPERACION DE significancia.py")
    print("=" * 78)
    cfg = ept.configuraciones()["op"]
    for nivel in ("nivel1", "nivel2"):
        for nombre in sorted(cfg[nivel]):
            print(f"  {nivel} {nombre:<14} "
                  f"{ept.etiqueta(cfg, nombre, nivel)}")
    print(f"  tolerancia de onset +-{cfg['tol_onset_s']:g} s | "
          f"IoU minimo {ed.IOU_MIN:g}\n")

    todo = ept.globales(cfg, a, catalogo, valida, horas, canales,
                        usar_cache)

    print(f"\n  Verificacion contra los valores citados")
    print(f"  {'metrica':<28}{'citado':>9}{'medido':>10}{'dif':>11}")
    peor = 0.0
    for nombre, obj in ept.OBJETIVO_N1.items():
        v = todo["nivel1"][nombre]["recall"]
        peor = max(peor, abs(v - obj))
        print(f"  {'nivel1 recall ' + nombre:<28}{obj:>9.3f}{v:>10.4f}"
              f"{v - obj:>+11.2e}")
    for met, obj in ept.OBJETIVO_N2_EQT.items():
        v = todo["nivel2"]["eqtransformer"][met]
        peor = max(peor, abs(v - obj))
        print(f"  {'nivel2 eqt ' + met:<28}{obj:>9.3f}{v:>10.4f}"
              f"{v - obj:>+11.2e}")
    ok = peor <= ept.TOL_REPRODUCE
    print(f"\n  Diferencia maxima {peor:.2e} -> "
          f"{'REPRODUCE' if ok else 'NO reproduce'} "
          f"(tolerancia {ept.TOL_REPRODUCE:.0e})")
    if not ok:
        return None

    banderas = {}
    for nombre in DETECTORES_N1:
        v, _, _ = sig.acertados_onset(catalogo, todo["onsets"][nombre])
        banderas[f"n1_{nombre}"] = v
    for nombre in DETECTORES_N2:
        v, _, _ = sig.acertados_iou(catalogo, todo["detecciones"][nombre],
                                    ed.IOU_MIN)
        banderas[f"n2_{nombre}"] = v
    return banderas


# =====================================================================
# REPORTES
# =====================================================================

def describir(nombre, v, ancho=30):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if v.size == 0:
        print(f"  {nombre:<{ancho}}{'sin razon valida':>10}")
        return
    print(f"  {nombre:<{ancho}}{v.size:>5}"
          f"{np.percentile(v, 10):>9.1f}{np.median(v):>9.1f}"
          f"{np.percentile(v, 90):>9.1f}{v.mean():>9.1f}{v.std():>8.1f}")


def comparar_variantes(df, canales):
    """Cuantos eventos cambian entre la variante base y la estricta."""
    print("\n" + "=" * 78)
    print("  BASE CONTRA ESTRICTA")
    print("=" * 78)
    print("  base      ventana de ruido completa de 60 s")
    print("  estricta  se sacan del ruido las muestras de otro evento del")
    print(f"            catalogo y las iguales a cero; se exigen "
          f"{MIN_RUIDO_S:g} s limpios\n")

    b, e = df.snr_base_db.to_numpy(), df.snr_est_db.to_numpy()
    fin_b, fin_e = np.isfinite(b), np.isfinite(e)
    print(f"  {'':<34}{'eventos':>9}")
    print(f"  {'con razon en base':<34}{int(fin_b.sum()):>9}")
    print(f"  {'con razon en estricta':<34}{int(fin_e.sum()):>9}")
    print(f"  {'sin razon en base':<34}{int((~fin_b).sum()):>9}")
    print(f"  {'sin razon en estricta':<34}{int((~fin_e).sum()):>9}")
    print(f"  {'pierden la razon al ser estricta':<34}"
          f"{int((fin_b & ~fin_e).sum()):>9}")

    amb = fin_b & fin_e
    d = e[amb] - b[amb]
    cambian = np.abs(d) > TOL_CAMBIO_DB
    print(f"\n  De los {int(amb.sum())} eventos con razon en las dos:")
    print(f"    cambian mas de {TOL_CAMBIO_DB:g} dB : {int(cambian.sum())}"
          f"  ({100 * cambian.mean():.1f} %)")
    if cambian.any():
        dc = d[cambian]
        print(f"    cambio (estricta - base): mediana {np.median(dc):+.2f} dB"
              f" | p10 {np.percentile(dc, 10):+.2f} | "
              f"p90 {np.percentile(dc, 90):+.2f} | max |.| "
              f"{np.abs(dc).max():.2f}")
        print(f"    suben {int((dc > 0).sum())}  bajan {int((dc < 0).sum())}")
        print("\n    Los 10 cambios mas grandes")
        print(f"    {'ev':>4}{'clase':>7}{'tramo':>7}{'base':>9}"
              f"{'estricta':>10}{'dif':>9}")
        sub = df[amb].copy()
        sub["dif"] = d
        for _, r in sub.reindex(sub.dif.abs().sort_values(
                ascending=False).index).head(10).iterrows():
            print(f"    {int(r.evento):>4}{r.clase:>7}{int(r.tramo):>7}"
                  f"{r.snr_base_db:>9.1f}{r.snr_est_db:>10.1f}"
                  f"{r.dif:>+9.1f}")

    print(f"\n  Canales con dato por evento (de {len(canales)} utilizables)")
    print(f"    base     mediana {df.n_canales_base.median():.0f} | "
          f"minimo {int(df.n_canales_base.min())} | "
          f"eventos con 0 canales {int((df.n_canales_base == 0).sum())}")
    print(f"    estricta mediana {df.n_canales_est.median():.0f} | "
          f"minimo {int(df.n_canales_est.min())} | "
          f"eventos con 0 canales {int((df.n_canales_est == 0).sum())}")


def distribuciones(df):
    """Distribucion de la razon por clase y por tramo."""
    for etiqueta, col in (("ESTRICTA", "snr_est_db"), ("BASE", "snr_base_db")):
        print("\n" + "=" * 78)
        print(f"  DISTRIBUCION DE LA RAZON - VARIANTE {etiqueta}")
        print("=" * 78)
        v = df[col].to_numpy()
        print(f"  {'grupo':<30}{'n':>5}{'p10':>9}{'mediana':>9}"
              f"{'p90':>9}{'media':>9}{'sd':>8}   dB")
        describir("todos", v)
        print()
        for c in ed.CLASES:
            describir(f"clase {c}", df[df.clase == c][col].to_numpy())
        print()
        for t in sorted(df.tramo.unique()):
            describir(f"tramo {t}", df[df.tramo == t][col].to_numpy())
        print()
        for t in sorted(df.tramo.unique()):
            for c in ed.CLASES:
                sub = df[(df.tramo == t) & (df.clase == c)]
                if len(sub):
                    describir(f"tramo {t} clase {c}", sub[col].to_numpy())


def sensibilidad_por_rango(df, banderas):
    """Sensibilidad de cada detector por rango de razon."""
    for etiqueta, col in (("ESTRICTA", "snr_est_db"),
                          ("BASE", "snr_base_db")):
        print("\n" + "=" * 78)
        print(f"  SENSIBILIDAD POR RANGO DE RAZON - VARIANTE {etiqueta}")
        print("=" * 78)
        print("  Cada celda es la fraccion de eventos de ese rango que el")
        print("  detector encuentra, con n = eventos del rango.\n")
        v = df[col].to_numpy()
        claves = [f"n1_{d}" for d in DETECTORES_N1] \
            + [f"n2_{d}" for d in DETECTORES_N2]
        enc = f"  {'rango dB':<12}{'n':>5}"
        for k in claves:
            enc += f"{k:>18}"
        print(enc)
        for lo, hi, nombre in RANGOS:
            m = np.isfinite(v) & (v >= lo) & (v < hi)
            linea = f"  {nombre:<12}{int(m.sum()):>5}"
            for k in claves:
                linea += (f"{df[k].to_numpy()[m].mean():>18.3f}"
                          if m.sum() else f"{'-':>18}")
            print(linea)
        m = ~np.isfinite(v)
        linea = f"  {'sin razon':<12}{int(m.sum()):>5}"
        for k in claves:
            linea += (f"{df[k].to_numpy()[m].mean():>18.3f}"
                      if m.sum() else f"{'-':>18}")
        print(linea)
        linea = f"  {'TODOS':<12}{len(df):>5}"
        for k in claves:
            linea += f"{df[k].to_numpy().mean():>18.3f}"
        print(linea)

        print(f"\n  Razon de los encontrados contra los perdidos "
              f"(variante {etiqueta.lower()})")
        print(f"  {'detector':<18}{'n enc':>7}{'med enc':>10}"
              f"{'n perd':>8}{'med perd':>10}{'dif':>8}")
        for k in claves:
            b = df[k].to_numpy()
            enc_v = v[b & np.isfinite(v)]
            per_v = v[(~b) & np.isfinite(v)]
            if enc_v.size == 0 or per_v.size == 0:
                print(f"  {k:<18}{enc_v.size:>7}{'-':>10}"
                      f"{per_v.size:>8}{'-':>10}{'-':>8}")
                continue
            me, mp = float(np.median(enc_v)), float(np.median(per_v))
            print(f"  {k:<18}{enc_v.size:>7}{me:>10.1f}"
                  f"{per_v.size:>8}{mp:>10.1f}{me - mp:>+8.1f}")


def margen_sobre_piso(df, piso):
    """
    Razon medida de cada clase contra el piso que impone su duracion.

    El piso sale del caso 3 de la validacion sintetica y se interpola a
    la duracion mediana de cada clase. Es el valor que reportaria el
    estimador si el evento fuera puro ruido gaussiano de esa duracion.
    El margen es cuanto sobresale la clase por encima de ese piso.

    CUIDADO: el piso se midio con ruido gaussiano blanco. El ruido
    sismico de fondo esta correlacionado y no es gaussiano, asi que el
    piso real puede ser distinto. Esto es una referencia, no el umbral
    de deteccion de nada.
    """
    if not piso:
        return
    xs = np.array(sorted(piso))
    ys = np.array([piso[x] for x in xs])
    print("\n" + "=" * 78)
    print("  RAZON MEDIDA CONTRA EL PISO QUE IMPONE LA DURACION")
    print("=" * 78)
    print("  El piso viene del caso 3 (ruido gaussiano) interpolado a la")
    print("  duracion mediana de cada grupo. Margen = mediana - piso.")
    print("  Un margen cercano a cero significa que, con este estimador y")
    print("  sobre la traza sin filtrar, ese grupo no se distingue de un")
    print("  tramo de ruido de la misma duracion.\n")
    print(f"  {'grupo':<16}{'n':>5}{'dur med_s':>11}{'piso_dB':>10}"
          f"{'mediana_dB':>12}{'margen_dB':>11}")
    grupos = [("clase " + c, df[df.clase == c]) for c in ed.CLASES]
    grupos += [(f"tramo {t}", df[df.tramo == t])
               for t in sorted(df.tramo.unique())]
    grupos += [("todos", df)]
    for nombre, sub in grupos:
        v = sub.snr_est_db.to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        d_med = float(sub.dur_s.median())
        p = float(np.interp(d_med, xs, ys))
        m = float(np.median(v))
        print(f"  {nombre:<16}{v.size:>5}{d_med:>11.1f}{p:>10.2f}"
              f"{m:>12.1f}{m - p:>+11.1f}")


def sin_ruido_valido(df, canales):
    """Detalle de los eventos que quedaron sin razon."""
    print("\n" + "=" * 78)
    print("  EVENTOS SIN RAZON VALIDA")
    print("=" * 78)
    for etiqueta, col in (("base", "snr_base_db"), ("estricta", "snr_est_db")):
        sub = df[~np.isfinite(df[col])]
        print(f"\n  Variante {etiqueta}: {len(sub)} de {len(df)} eventos")
        if sub.empty:
            continue
        print(f"    por clase: "
              + ", ".join(f"{c}:{int((sub.clase == c).sum())}"
                          for c in ed.CLASES
                          if int((sub.clase == c).sum())))
        print(f"    por tramo: "
              + ", ".join(f"{t}:{int((sub.tramo == t).sum())}"
                          for t in sorted(df.tramo.unique())
                          if int((sub.tramo == t).sum())))
        print(f"    {'ev':>4}{'clase':>7}{'tramo':>7}{'ini':>12}"
              f"{'dur_s':>9}{'canales':>9}")
        ncol = "n_canales_base" if etiqueta == "base" else "n_canales_est"
        for _, r in sub.head(25).iterrows():
            print(f"    {int(r.evento):>4}{r.clase:>7}{int(r.tramo):>7}"
                  f"{int(r.ini):>12}{r.dur_s:>9.1f}{int(r[ncol]):>9}")
        if len(sub) > 25:
            print(f"    ... y {len(sub) - 25} mas")


# =====================================================================
# MAIN
# =====================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--canales", nargs="+", type=int, default=ed.CANALES)
    ap.add_argument("--sin-cache", action="store_true",
                    help="recalcula STA/LTA en vez de leer su cacha")
    ap.add_argument("--solo-autotest", action="store_true")
    args = ap.parse_args(argv)

    ok_test, piso = autotest()
    if not ok_test:
        print("\n  La validacion sintetica no pasa. Me detengo: no mido")
        print("  sobre datos reales con una formula que no reproduce un")
        print("  caso de solucion conocida.")
        return 1
    if args.solo_autotest:
        return 0

    print("\nCargando traza y catalogo ...")
    a, catalogo, valida, horas = ed.cargar()
    n = a.shape[-1]
    print(f"  {len(catalogo)} eventos, {horas:.2f} h validas, "
          f"canales {args.canales}")

    banderas = banderas_deteccion(a, catalogo, valida, horas, args.canales,
                                  not args.sin_cache)
    if banderas is None:
        print("\n  No reproduzco los valores citados. Me detengo.")
        return 1

    print("\nCalculando la razon de los 203 eventos ...", flush=True)
    ocupado, n_mascara = mascara_eventos(n)
    print(f"  Mascara de 'otro evento' construida con los {n_mascara} "
          f"eventos del CSV completo,")
    print(f"  no solo con los {len(catalogo)} que quedan tras excluir las "
          f"guardas de union.")
    df = razones(a, catalogo, args.canales, ocupado)

    segs = ed.segmentos(n)
    df["ini"] = [ev.ini for ev in catalogo]
    df["fin"] = [ev.fin for ev in catalogo]
    df["clase"] = [ev.clase for ev in catalogo]
    df["dur_s"] = [ev.dur / FS for ev in catalogo]
    df["tramo"] = [next(i for i, (s, e) in enumerate(segs, start=1)
                        if s <= ev.ini < e) for ev in catalogo]
    for k, v in banderas.items():
        df[k] = v

    cols = (["evento", "clase", "tramo", "ini", "fin", "dur_s",
             "snr_base_db", "snr_est_db", "n_canales_base", "n_canales_est"]
            + [f"snr_base_ch{c}" for c in args.canales]
            + [f"snr_est_ch{c}" for c in args.canales]
            + [f"n1_{d}" for d in DETECTORES_N1]
            + [f"n2_{d}" for d in DETECTORES_N2])
    df = df[cols]

    comparar_variantes(df, args.canales)
    distribuciones(df)
    sensibilidad_por_rango(df, banderas)
    margen_sobre_piso(df, piso)
    sin_ruido_valido(df, args.canales)

    os.makedirs(SALIDA, exist_ok=True)
    df.to_csv(CSV_SALIDA, index=False)
    print(f"\nTabla de {len(df)} eventos en {CSV_SALIDA}")

    print("\n" + "=" * 78)
    print("  LIMITES DE ESTA MEDICION")
    print("=" * 78)
    print("""  La razon se calcula sobre la traza SIN filtrar, igual que en
  revisar_falsos_positivos.py. Los detectores no trabajan sobre esa
  senal: STA/LTA filtra entre 1 y 20 Hz y los modelos normalizan cada
  ventana. Una razon baja aqui no implica que el detector vea poco, ni
  al reves.

  Los casos 2 y 3 de la validacion sintetica miden dos sesgos del
  estimador. Primero, cuando el evento y el ruido son comparables el
  pico sobre RMS sobreestima la razon verdadera. Segundo, y mas
  importante, el estimador tiene un PISO que crece con la duracion del
  evento, porque el pico de un tramo largo de puro ruido es mayor que
  el de uno corto. Los dos sesgos empujan hacia arriba, asi que el
  rango '< 5 dB' se vacia por construccion y la comparacion entre
  clases de duracion distinta (TR contra VT, por ejemplo) esta
  confundida con la duracion. El caso 3 da la magnitud exacta de esa
  confusion para las duraciones del catalogo.

  Este script no elige umbrales ni propone cambios de configuracion.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
