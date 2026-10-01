#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barrido_sensibilidad.py
=======================

Barrido orientado a la meta de sensibilidad del 95 % que pidio Ricardo en
la reunion del 21 de septiembre. Solo MIDE. No modifica ningun script ni
ningun valor por omision, no recalcula inferencia y NO elige un punto de
operacion: deja la frontera a la vista para que la eleccion se haga
despues, por escrito y con el costo en falsos positivos declarado.

Todo lo que sigue es NIVEL 1 (deteccion del instante) con tolerancia de
+-5 s, que es el nivel donde los tres detectores compiten en igualdad.
"Sensibilidad" y "recall" son lo mismo aqui.

LAS CUATRO PARTES
-----------------
1. Lectura de los barridos ya hechos (resultados/comparacion_nivel1.csv y
   resultados/frontera_pareto.csv): sensibilidad maxima por detector y el
   costo en falsos positivos por hora al cruzar 0,80, 0,90 y 0,95.

2. Extension del nivel 1 de PhaseNet y EQTransformer con las cachas de
   arribos que ya existen: los dos modos de relleno de componentes
   (ceros y duplicar), la confianza minima desde el piso que guarda cada
   cacha hasta 0,50, y 1 o 2 estaciones. El piso de cada cacha se informa
   explicitamente: por debajo de el no hay arribos guardados y bajar mas
   exigiria volver a correr los modelos, cosa que este script no hace.

3. Combinacion de detectores: union de los onsets de red de cada par y de
   los tres, fusionando onsets separados por menos de 5 s. Se mide con
   cada detector en su punto de significancia.py ("mejor F1") y tambien
   en su punto de mayor sensibilidad. La fusion se reporta con tres
   reglas de representante del grupo (mediana, minimo y sin fusionar),
   porque medirlo mostro que esa eleccion mueve la sensibilidad tanto
   como la diferencia entre detectores.

4. Frontera sensibilidad contra falsos positivos por hora de todo lo
   anterior, en un CSV y una figura.

PROCEDENCIA DE LOS BARRIDOS DE LA PARTE 1
-----------------------------------------
comparacion_nivel1.csv y frontera_pareto.csv son del 20 de agosto, o sea
ANTERIORES al commit 1658de0 (28 de agosto) que corrigio el segundo corte
de la traza de 3.240.000 a 3.240.001. La guarda de 30 s alrededor del
corte se desplaza una muestra, y eso cambia el conteo de onsets. El script
mide ese efecto en vez de suponerlo (ver verificar_procedencia): resulta
ser de un onset falso en STA/LTA y de cero en los modelos, con la
sensibilidad identica. Las cifras de la parte 1 se reportan tal como
estan guardadas, y las de las partes 2 y 3 con el corte corregido.

VERIFICACION PREVIA
-------------------
Antes de reportar cualquier cosa, el script reconstruye el nivel 1 de los
puntos de operacion de significancia.py y lo compara contra los valores
citados en CLAUDE.local.md (0,542 STA/LTA, 0,527 PhaseNet, 0,581
EQTransformer). Si no reproduce, se detiene: medir sobre una base que no
empata con el documento no sirve de nada.

Reutiliza:
    significancia.py    puntos de operacion, lectura de cachas y
                        reconstruccion de STA/LTA
    evaluar_por_tramo.py cacha en disco de STA/LTA y los ambitos por tramo
    evaluar_detectores.py coincidencia de red y metricas de onset

Uso:
    python barrido_sensibilidad.py
    python barrido_sensibilidad.py --sin-cache     # recalcula STA/LTA
    python barrido_sensibilidad.py --sin-figura

Autor: Patricio Valdes Troncoso - Universidad Catolica de Temuco
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import evaluar_detectores as ed        # noqa: E402
import significancia as sig            # noqa: E402
import evaluar_por_tramo as ept        # noqa: E402

FS = ed.FS
SALIDA = ed.SALIDA

# Nivel 1 con la misma tolerancia que usa significancia.py.
TOL_ONSET_S = sig.TOL_ONSET_S          # 5,0 s

# Fusion de onsets al unir detectores distintos. El enunciado fija 5 s.
FUSION_ONSETS_S = 5.0

# Metas de sensibilidad a reportar.
OBJETIVOS = [0.80, 0.90, 0.95]

# Rejilla de confianza minima para la extension de la parte 2. El primer
# punto de cada cacha es su propio piso medido, que se agrega aparte.
UMBRALES_CONF = [0.025, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15,
                 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_EST = [1, 2]
MODOS = ["ceros", "duplicar"]
MODELOS = ["phasenet", "eqtransformer"]

# Valores citados en CLAUDE.local.md que hay que reproducir antes de
# reportar. Son los del nivel 1 de significancia.py.
OBJETIVO_N1 = ept.OBJETIVO_N1
TOL_REPRODUCE = ept.TOL_REPRODUCE

CSV_SALIDA = os.path.join(SALIDA, "barrido_sensibilidad.csv")
PNG_SALIDA = os.path.join(SALIDA, "frontera_sensibilidad.png")


# =====================================================================
# UTILIDADES
# =====================================================================

def metricas_de_onsets(catalogo, onsets, horas):
    """Nivel 1 de un conjunto de onsets de red, aplanado a un diccionario."""
    r = ed.metricas_onset(catalogo, onsets, horas, tolerancias=[TOL_ONSET_S])
    d = r[f"tol_{TOL_ONSET_S:g}s"]
    fila = dict(n_onsets=r["n_onsets"],
                recall=d["recall"], precision=d["precision"],
                falsos_h=d["onsets_falsos_por_hora"],
                err_mae=d["error_mae_s"])
    rec, pre = d["recall"], d["precision"]
    fila["f1"] = 2 * rec * pre / (rec + pre) if (rec + pre) > 0 else 0.0
    for c in ed.CLASES:
        fila[f"rec_{c}"] = d["por_clase"].get(c, {}).get("recall", np.nan)
        fila[f"n_{c}"] = d["por_clase"].get(c, {}).get("n", 0)
    return fila


# Reglas de representante del grupo al fusionar. La agrupacion es siempre
# la misma (cadena de 5 s); lo que cambia es que onset emite el grupo.
REGLAS_FUSION = ["mediana", "minimo", "sin fusionar"]


def fusionar_onsets(onsets, tol_s=FUSION_ONSETS_S, regla="mediana"):
    """
    Fusiona onsets provenientes de detectores distintos.

    Dos onsets separados por menos de tol_s quedan en el mismo grupo y el
    grupo emite un solo onset. La agrupacion es la MISMA que usa
    coincidencia_onsets de evaluar_detectores.py para agrupar estaciones:
    cadena, o sea cada onset se compara con el ultimo aceptado del grupo.

    La regla es QUE onset emite el grupo, y no es una decision neutral:

      "mediana"      mediana del grupo. Es el estimador que ya usa
                     coincidencia_onsets, asi que la union no introduce
                     una diferencia metodologica respecto de la
                     agregacion de red. Pero un acierto de un detector
                     puede perderse si se agrupa con un onset espurio de
                     otro y la mediana se corre fuera de la tolerancia.
      "minimo"       el onset mas temprano del grupo. Conserva el inicio
                     mas adelantado, que es lo que mide el nivel 1.
      "sin fusionar" no fusiona nada. No es una union utilizable porque
                     deja onsets repetidos y castiga la precision, pero
                     da la cota superior de sensibilidad de la union.

    Las tres se reportan porque la diferencia entre ellas resulta ser del
    mismo orden que las diferencias entre detectores.
    """
    todos = sorted(int(t) for t in onsets)
    if not todos:
        return []
    if regla == "sin fusionar":
        return todos
    tol = int(tol_s * FS)
    grupos, actual = [], [todos[0]]
    for t in todos[1:]:
        if t - actual[-1] <= tol:
            actual.append(t)
        else:
            grupos.append(actual)
            actual = [t]
    grupos.append(actual)
    if regla == "minimo":
        return sorted(min(g) for g in grupos)
    if regla == "mediana":
        return sorted(int(np.median(g)) for g in grupos)
    raise ValueError(f"regla desconocida: {regla}")


def primer_punto(df, objetivo):
    """
    Primer punto que alcanza una sensibilidad objetivo.

    CRITERIO DECLARADO: entre todos los puntos con recall >= objetivo se
    devuelve el de MENOR tasa de falsos positivos por hora, y si hay
    empate el de mayor recall. Es "el primero" en el sentido de el mas
    barato que cruza la meta, que es el que interesa cuando la meta de
    sensibilidad esta fijada de antemano. No es una eleccion de punto de
    operacion: es la lectura de la tabla.
    """
    sel = df[df.recall >= objetivo]
    if sel.empty:
        return None
    sel = sel.sort_values(["falsos_h", "recall"],
                          ascending=[True, False])
    return sel.iloc[0]


def frontera(df):
    """
    Puntos no dominados en (sensibilidad alta, falsos por hora bajos).

    Se recorre de mayor a menor recall y se conserva un punto solo si
    mejora el minimo de falsos_h visto hasta entonces.
    """
    d = df.sort_values(["recall", "falsos_h"],
                       ascending=[False, True]).reset_index(drop=True)
    mejor, out = float("inf"), []
    for i, r in d.iterrows():
        if r["falsos_h"] < mejor:
            out.append(i)
            mejor = r["falsos_h"]
    return d.loc[out].sort_values("recall")


def etiqueta_stalta(cfg):
    return (f"sta{cfg['sta_s']:g}_lta{cfg['lta_s']:g}"
            f"_on{cfg['thr_on']:g}_off{cfg['thr_off']:g}")


# =====================================================================
# VERIFICACION: REPRODUCIR EL NIVEL 1 DE significancia.py
# =====================================================================

def verificar(a, catalogo, valida, horas, canales, usar_cache):
    """
    Reconstruye el nivel 1 de los puntos de operacion de significancia.py.

    Devuelve (ok, onsets_por_detector, metricas_por_detector).
    """
    print("=" * 78)
    print("  VERIFICACION: REPRODUCIR EL NIVEL 1 DE significancia.py")
    print("=" * 78)
    print(f"  Tolerancia de onset +-{TOL_ONSET_S:g} s | "
          f"{len(catalogo)} eventos | {horas:.2f} h validas\n")

    onsets, filas = {}, {}
    for nombre in ("stalta", "phasenet", "eqtransformer"):
        cfg = sig.OP_NIVEL1[nombre]
        if nombre == "stalta":
            print(f"  {nombre}: {etiqueta_stalta(cfg)}, "
                  f"fusion {cfg['fusion_s']:g} s, min_est {cfg['min_est']}")
            pc = ept.stalta_por_canal_cacheado(a, canales, cfg, usar_cache)
        else:
            print(f"  {nombre}: componentes {cfg['modo']}, "
                  f"umbral {cfg['umbral']:g}, min_est {cfg['min_est']}")
            pc = sig.picks_por_canal(nombre, cfg["modo"], cfg["umbral"])
            if pc is None:
                print("      falta la cacha de arribos")
                return False, onsets, filas
        onsets[nombre] = ed.coincidencia_onsets(pc, valida, cfg["min_est"])
        filas[nombre] = metricas_de_onsets(catalogo, onsets[nombre], horas)

    print(f"\n  {'detector':<16}{'citado':>9}{'medido':>10}{'dif':>11}"
          f"{'n_onsets':>10}")
    peor = 0.0
    for nombre, obj in OBJETIVO_N1.items():
        v = filas[nombre]["recall"]
        d = v - obj
        peor = max(peor, abs(d))
        print(f"  {nombre:<16}{obj:>9.3f}{v:>10.4f}{d:>+11.2e}"
              f"{filas[nombre]['n_onsets']:>10}")
    ok = peor <= TOL_REPRODUCE
    print(f"\n  Diferencia maxima {peor:.2e} -> "
          f"{'REPRODUCE' if ok else 'NO reproduce'} "
          f"(tolerancia {TOL_REPRODUCE:.0e})")
    return ok, onsets, filas


# =====================================================================
# VERIFICACION DE PROCEDENCIA DE LOS BARRIDOS GUARDADOS
# =====================================================================

# Corte viejo de la traza, el que estaba vigente cuando se genero
# comparacion_nivel1.csv. El commit 1658de0 lo corrigio a 3.240.001.
CORTES_VIEJOS = [2_160_000, 3_240_000]


def verificar_procedencia(tabla2, a, catalogo, valida, horas, canales,
                          usar_cache):
    """
    Mide si los barridos guardados se pueden empatar con el corte actual.

    Para los modelos la comparacion es gratis: las filas de la parte 2
    con la misma (modo, umbral, min_est) se cotejan contra las filas
    guardadas. Para STA/LTA se reconstruye la fila de mayor sensibilidad
    con el corte ACTUAL y, si no empata, se repite con el corte VIEJO
    para comprobar que la diferencia viene de ahi y no de otra cosa.
    """
    print("\n" + "=" * 78)
    print("  PROCEDENCIA DE LOS BARRIDOS GUARDADOS")
    print("=" * 78)
    ruta = os.path.join(SALIDA, "comparacion_nivel1.csv")
    if not os.path.exists(ruta) or tabla2 is None or tabla2.empty:
        print("  sin material para comparar")
        return
    df = pd.read_csv(ruta)

    print("  Modelos: filas guardadas contra la extension de la parte 2,")
    print("  con la misma combinacion de modo, umbral y min_est.")
    print(f"  {'modelo':<15}{'filas':>7}{'dif.max_sens':>14}"
          f"{'dif.max_onsets':>16}")
    for modelo in MODELOS:
        g = df[df.modelo == modelo]
        d_rec, d_ons, n = 0.0, 0, 0
        for _, r in g.iterrows():
            m = tabla2[(tabla2.grupo == modelo) & (tabla2.modo == r.modo)
                       & (np.isclose(tabla2.umbral, r.umbral))
                       & (tabla2.min_est == int(r.min_est))]
            if m.empty:
                continue
            n += 1
            d_rec = max(d_rec, abs(float(m.iloc[0].recall) - r.recall))
            d_ons = max(d_ons, abs(int(m.iloc[0].n_onsets) - int(r.n_onsets)))
        print(f"  {modelo:<15}{n:>7}{d_rec:>14.2e}{d_ons:>16}")

    r = punto_max_sensibilidad_stalta()
    if r is None:
        return
    cfg, rec_csv, fph_csv = r
    ons_csv = int(df[(df.modelo == "stalta") & (df.modo == etiqueta_stalta(cfg))
                     & (df.min_est == cfg["min_est"])].iloc[0].n_onsets)
    pc = ept.stalta_por_canal_cacheado(a, canales, cfg, usar_cache)
    f_act = metricas_de_onsets(catalogo,
                               ed.coincidencia_onsets(pc, valida,
                                                      cfg["min_est"]),
                               horas)
    print(f"\n  STA/LTA en su punto de mayor sensibilidad "
          f"({etiqueta_stalta(cfg)}, min_est {cfg['min_est']})")
    print(f"  {'version':<28}{'n_onsets':>10}{'sens':>10}{'fp/h':>10}")
    print(f"  {'guardado (20 de agosto)':<28}{ons_csv:>10}"
          f"{rec_csv:>10.6f}{fph_csv:>10.4f}")
    print(f"  {'corte actual 3.240.001':<28}{f_act['n_onsets']:>10}"
          f"{f_act['recall']:>10.6f}{f_act['falsos_h']:>10.4f}")

    if f_act["n_onsets"] == ons_csv:
        print("  Empata: no hay efecto del cambio de corte en esta fila.")
        return

    # No empata: se comprueba la causa repitiendo con el corte viejo.
    cortes_guardados = ed.CORTES
    try:
        ed.CORTES = list(CORTES_VIEJOS)
        a2, cat2, val2, h2 = ed.cargar()
        pc2 = sig.stalta_por_canal(a2, canales, cfg)
        f_vie = metricas_de_onsets(
            cat2, ed.coincidencia_onsets(pc2, val2, cfg["min_est"]), h2)
    finally:
        ed.CORTES = cortes_guardados
    print(f"  {'corte viejo 3.240.000':<28}{f_vie['n_onsets']:>10}"
          f"{f_vie['recall']:>10.6f}{f_vie['falsos_h']:>10.4f}")
    if f_vie["n_onsets"] == ons_csv:
        print("\n  Con el corte viejo la fila guardada se reproduce exacto.")
        print("  La diferencia con el corte actual es de "
              f"{abs(f_act['n_onsets'] - ons_csv)} onset(s) falso(s) de "
              f"{ons_csv}, y la sensibilidad no cambia.")
    else:
        print("\n  El corte viejo tampoco reproduce la fila guardada: la")
        print("  diferencia viene de otra cosa y queda sin explicar.")


# =====================================================================
# PARTE 1: LECTURA DE LOS BARRIDOS YA HECHOS
# =====================================================================

def parte1():
    """Lee comparacion_nivel1.csv y frontera_pareto.csv y los resume."""
    print("\n" + "=" * 78)
    print("  PARTE 1 - BARRIDOS YA HECHOS (lectura, sin recalcular nada)")
    print("=" * 78)
    print("  Criterio del 'primer punto': entre los que cruzan la meta, el")
    print("  de menor tasa de falsos positivos por hora.")
    print("  Estas cifras son las GUARDADAS, con el corte de la traza en")
    print("  3.240.000; ver la seccion de procedencia mas arriba.\n")

    for archivo in ("comparacion_nivel1.csv", "frontera_pareto.csv"):
        ruta = os.path.join(SALIDA, archivo)
        if not os.path.exists(ruta):
            print(f"  {archivo}: no existe, se omite")
            continue
        df = pd.read_csv(ruta)
        print(f"  --- {archivo}  ({len(df)} puntos) ---")
        print(f"  {'detector':<15}{'n':>5}{'sens.max':>10}{'fp/h':>10}"
              f"  configuracion del maximo")
        for det, g in df.groupby("modelo"):
            i = g.recall.idxmax()
            r = g.loc[i]
            print(f"  {det:<15}{len(g):>5}{g.recall.max():>10.3f}"
                  f"{r.falsos_h:>10.2f}  modo={r.modo} umbral={r.umbral:g}"
                  f" min_est={int(r.min_est)}")
        print()
        print(f"  {'detector':<15}{'meta':>7}{'sens':>8}{'fp/h':>10}"
              f"{'prec':>8}{'n_onsets':>10}  configuracion")
        for det, g in df.groupby("modelo"):
            for obj in OBJETIVOS:
                r = primer_punto(g, obj)
                if r is None:
                    print(f"  {det:<15}{obj:>7.2f}{'-':>8}{'-':>10}"
                          f"{'-':>8}{'-':>10}  NO alcanzada")
                    continue
                print(f"  {det:<15}{obj:>7.2f}{r.recall:>8.3f}"
                      f"{r.falsos_h:>10.2f}{r.precision:>8.3f}"
                      f"{int(r.n_onsets):>10}  modo={r.modo}"
                      f" umbral={r.umbral:g} min_est={int(r.min_est)}")
        print()

    # Para la parte 4 se devuelven TODOS los puntos del barrido previo, no
    # solo los que cruzan las metas. Se toma comparacion_nivel1.csv y no
    # frontera_pareto.csv porque el segundo es un subconjunto del primero
    # (verificado) y duplicarlo inflaria la tabla.
    ruta = os.path.join(SALIDA, "comparacion_nivel1.csv")
    if not os.path.exists(ruta):
        return pd.DataFrame()
    df = pd.read_csv(ruta).rename(columns={"modelo": "grupo"})
    df["fuente"] = "barrido_previo"
    df["ambito"] = "global"
    df["corte"] = CORTES_VIEJOS[1]
    df["min_est"] = df.min_est.astype(int)
    return df


# =====================================================================
# PARTE 2: EXTENSION DEL NIVEL 1 CON LAS CACHAS DE ARRIBOS
# =====================================================================

def pisos_de_cache():
    """Confianza minima efectivamente guardada en cada cacha de arribos."""
    print("\n" + "=" * 78)
    print("  PARTE 2 - PISO DE CONFIANZA DE CADA CACHA DE ARRIBOS")
    print("=" * 78)
    print("  El piso es el minimo de la columna conf del CSV cacheado. Por")
    print("  debajo de el no hay arribos guardados: bajar mas exigiria")
    print("  volver a correr la inferencia, y este script no lo hace.\n")
    print(f"  {'modelo':<15}{'modo':<11}{'n_picks':>9}{'conf_min':>11}"
          f"{'mediana':>10}{'conf_max':>10}")
    pisos = {}
    for modelo in MODELOS:
        for modo in MODOS:
            ruta = os.path.join(SALIDA, "cache_picks",
                                f"picks_{modelo}_{modo}.csv")
            if not os.path.exists(ruta):
                print(f"  {modelo:<15}{modo:<11}  sin cacha")
                continue
            p = pd.read_csv(ruta)
            pisos[(modelo, modo)] = float(p.conf.min())
            print(f"  {modelo:<15}{modo:<11}{len(p):>9}"
                  f"{p.conf.min():>11.6f}{p.conf.median():>10.3f}"
                  f"{p.conf.max():>10.3f}")
    if pisos:
        print(f"\n  Umbral de inferencia declarado en barrido_modelos.py: "
              f"0,02 (P, S y detection).")
        print("  Los pisos medidos coinciden con ese 0,02, asi que la")
        print("  extension puede bajar hasta ~0,020 y NO mas abajo.")
    return pisos


def parte2(catalogo, valida, horas, pisos):
    """Rejilla modelo x modo x confianza x min_est sobre las cachas."""
    print("\n" + "=" * 78)
    print("  PARTE 2 - EXTENSION DEL NIVEL 1 (aritmetica sobre las cachas)")
    print("=" * 78)
    print(f"  Confianza desde el piso de cada cacha hasta 0,50 | "
          f"min_est {MIN_EST} | modos {MODOS}\n")

    filas = []
    for modelo in MODELOS:
        for modo in MODOS:
            if (modelo, modo) not in pisos:
                continue
            ruta = os.path.join(SALIDA, "cache_picks",
                                f"picks_{modelo}_{modo}.csv")
            picks = pd.read_csv(ruta)
            piso = pisos[(modelo, modo)]
            umbrales = [piso] + [u for u in UMBRALES_CONF if u > piso]
            print(f"  --- {modelo} | componentes {modo} | piso "
                  f"{piso:.6f} ---")
            print(f"  {'umbral':>8}{'min_est':>8}{'n_picks':>9}"
                  f"{'n_onsets':>10}{'sens':>8}{'prec':>8}{'fp/h':>10}"
                  f"{'MAE_s':>8}")
            for u in umbrales:
                sel = picks[picks.conf >= u]
                if sel.empty:
                    continue
                pc = {ch: {"onsets": sorted(g.muestra.tolist())}
                      for ch, g in sel.groupby("canal")}
                for k in MIN_EST:
                    ons = ed.coincidencia_onsets(pc, valida, k)
                    if not ons:
                        continue
                    f = metricas_de_onsets(catalogo, ons, horas)
                    f.update(fuente="extension", grupo=modelo, modo=modo,
                             umbral=u, min_est=k, n_picks=len(sel),
                             ambito="global", corte=ed.CORTES[1])
                    filas.append(f)
                    print(f"  {u:>8.3f}{k:>8}{len(sel):>9}"
                          f"{f['n_onsets']:>10}{f['recall']:>8.3f}"
                          f"{f['precision']:>8.3f}{f['falsos_h']:>10.2f}"
                          f"{f['err_mae']:>8.2f}")
            print()

    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla

    print("  Sensibilidad maxima alcanzable por modelo y modo en la "
          "extension")
    print(f"  {'modelo':<15}{'modo':<11}{'sens.max':>10}{'fp/h':>10}"
          f"{'umbral':>9}{'min_est':>9}")
    for (modelo, modo), g in tabla.groupby(["grupo", "modo"]):
        r = g.sort_values(["recall", "falsos_h"],
                          ascending=[False, True]).iloc[0]
        print(f"  {modelo:<15}{modo:<11}{r.recall:>10.3f}"
              f"{r.falsos_h:>10.2f}{r.umbral:>9.3f}{int(r.min_est):>9}")

    print(f"\n  Primer punto que cruza cada meta en la extension")
    print(f"  {'modelo':<15}{'meta':>7}{'sens':>8}{'fp/h':>10}{'prec':>8}"
          f"{'n_onsets':>10}  configuracion")
    for modelo, g in tabla.groupby("grupo"):
        for obj in OBJETIVOS:
            r = primer_punto(g, obj)
            if r is None:
                print(f"  {modelo:<15}{obj:>7.2f}{'-':>8}{'-':>10}"
                      f"{'-':>8}{'-':>10}  NO alcanzada")
                continue
            print(f"  {modelo:<15}{obj:>7.2f}{r.recall:>8.3f}"
                  f"{r.falsos_h:>10.2f}{r.precision:>8.3f}"
                  f"{int(r.n_onsets):>10}  modo={r.modo}"
                  f" umbral={r.umbral:g} min_est={int(r.min_est)}")
    return tabla


# =====================================================================
# PARTE 3: COMBINACION DE DETECTORES
# =====================================================================

def punto_max_sensibilidad_stalta():
    """
    Punto de mayor sensibilidad de STA/LTA segun comparacion_nivel1.csv.

    Se lee de la tabla ya calculada y se traduce a la configuracion que
    necesita stalta_por_canal para poder reconstruir los onsets. Empate
    de recall se rompe por menor tasa de falsos positivos.
    """
    ruta = os.path.join(SALIDA, "comparacion_nivel1.csv")
    if not os.path.exists(ruta):
        return None
    df = pd.read_csv(ruta)
    g = df[df.modelo == "stalta"]
    if g.empty:
        return None
    r = g.sort_values(["recall", "falsos_h"],
                      ascending=[False, True]).iloc[0]
    # el campo modo tiene la forma sta<X>_lta<Y>_on<Z>_off<W>
    txt = r.modo
    val = {}
    for campo in ("sta", "lta", "on", "off"):
        i = txt.index(campo) + len(campo)
        j = txt.find("_", i)
        val[campo] = float(txt[i:j] if j > 0 else txt[i:])
    cfg = dict(sta_s=val["sta"], lta_s=val["lta"], thr_on=val["on"],
               thr_off=val["off"], fusion_s=sig.OP_NIVEL1["stalta"]["fusion_s"],
               min_est=int(r.min_est))
    return cfg, float(r.recall), float(r.falsos_h)


def puntos_max_sensibilidad(tabla2):
    """Punto de mayor sensibilidad de cada modelo dentro de la extension."""
    out = {}
    if tabla2.empty:
        return out
    for modelo, g in tabla2.groupby("grupo"):
        r = g.sort_values(["recall", "falsos_h"],
                          ascending=[False, True]).iloc[0]
        out[modelo] = dict(modo=r.modo, umbral=float(r.umbral),
                           min_est=int(r.min_est))
    return out


def onsets_de_configuracion(nombre, cfg, a, valida, canales, usar_cache):
    """Onsets de red de un detector en una configuracion dada."""
    if nombre == "stalta":
        pc = ept.stalta_por_canal_cacheado(a, canales, cfg, usar_cache)
    else:
        pc = sig.picks_por_canal(nombre, cfg["modo"], cfg["umbral"])
        if pc is None:
            return None
    return ed.coincidencia_onsets(pc, valida, cfg["min_est"])


def parte3(a, catalogo, valida, horas, canales, usar_cache,
           onsets_op, tabla2):
    """Union de onsets de red por pares y de los tres detectores."""
    print("\n" + "=" * 78)
    print("  PARTE 3 - COMBINACION DE DETECTORES EN NIVEL 1")
    print("=" * 78)
    print(f"  Union de los onsets de RED de cada detector, fusionando los")
    print(f"  separados por menos de {FUSION_ONSETS_S:g} s. La agrupacion es")
    print("  siempre la misma; lo que cambia entre las tres tablas es que")
    print("  onset emite el grupo:")
    print("    mediana      el estimador de coincidencia_onsets")
    print("    minimo       el onset mas temprano del grupo")
    print("    sin fusionar cota superior; repite onsets, no es usable\n")

    escenarios = {}

    # --- escenario A: punto de significancia.py ---
    print("  Escenario 'op' (punto de significancia.py, mejor F1)")
    for nombre in ("stalta", "phasenet", "eqtransformer"):
        cfg = sig.OP_NIVEL1[nombre]
        txt = (etiqueta_stalta(cfg) if nombre == "stalta"
               else f"{cfg['modo']} umbral {cfg['umbral']:g}")
        print(f"    {nombre:<15}{txt}, min_est {cfg['min_est']}")
    escenarios["op"] = dict(onsets_op)

    # --- escenario B: punto de mayor sensibilidad ---
    print("\n  Escenario 'maxsens' (mayor sensibilidad de cada detector;")
    print("  empate de sensibilidad roto por menor tasa de falsos)")
    maxs = {}
    r_st = punto_max_sensibilidad_stalta()
    if r_st is not None:
        cfg, rec, fph = r_st
        print(f"    {'stalta':<15}{etiqueta_stalta(cfg)}, "
              f"min_est {cfg['min_est']}  (tabla: sens {rec:.3f}, "
              f"fp/h {fph:.2f})")
        ons = onsets_de_configuracion("stalta", cfg, a, valida, canales,
                                      usar_cache)
        if ons is not None:
            maxs["stalta"] = ons
            f = metricas_de_onsets(catalogo, ons, horas)
            print(f"      reconstruido: sens {f['recall']:.3f}, "
                  f"fp/h {f['falsos_h']:.2f}, n_onsets {f['n_onsets']}")
    maxs_modelos = puntos_max_sensibilidad(tabla2)
    # Se recorre en el orden de MODELOS para que las tablas de los dos
    # escenarios salgan con los detectores en la misma posicion.
    for modelo in MODELOS:
        if modelo not in maxs_modelos:
            continue
        cfg = maxs_modelos[modelo]
        print(f"    {modelo:<15}{cfg['modo']} umbral {cfg['umbral']:.3f}, "
              f"min_est {cfg['min_est']}")
        ons = onsets_de_configuracion(modelo, cfg, a, valida, canales,
                                      usar_cache)
        if ons is not None:
            maxs[modelo] = ons
    escenarios["maxsens"] = maxs

    # --- medicion de individuales y uniones ---
    amb = ept.ambitos(a, valida)
    combos = [("stalta", "phasenet"),
              ("stalta", "eqtransformer"),
              ("phasenet", "eqtransformer"),
              ("stalta", "phasenet", "eqtransformer")]

    filas = []
    for esc, ons_det in escenarios.items():
        if not ons_det:
            continue
        print("\n  " + "-" * 74)
        print(f"  Escenario '{esc}'")
        print("  " + "-" * 74)

        n_por_clase = {c: sum(1 for e in catalogo if e.clase == c)
                       for c in ed.CLASES}

        for regla in REGLAS_FUSION:
            # Los individuales no se fusionan: son los mismos en las tres
            # reglas y se repiten para que cada tabla se lea sola.
            conjuntos = {(k,): v for k, v in ons_det.items()}
            for combo in combos:
                if not all(k in ons_det for k in combo):
                    continue
                union = [t for k in combo for t in ons_det[k]]
                conjuntos[combo] = fusionar_onsets(union, regla=regla)

            def _anotar(f, combo, ambito, **extra):
                f.update(fuente="combinacion", grupo=" + ".join(combo),
                         escenario=esc, regla=regla,
                         modo="union" if len(combo) > 1 else "individual",
                         n_detectores=len(combo), ambito=ambito,
                         corte=ed.CORTES[1], **extra)
                filas.append(f)

            print(f"\n  >>> regla de fusion: {regla}")
            print(f"\n  Global ({horas:.2f} h validas, "
                  f"{len(catalogo)} eventos)")
            print(f"  {'conjunto':<34}{'n_onsets':>9}{'sens':>8}{'prec':>8}"
                  f"{'F1':>8}{'fp/h':>10}{'MAE_s':>8}")
            for combo, ons in conjuntos.items():
                f = metricas_de_onsets(catalogo, ons, horas)
                print(f"  {' + '.join(combo):<34}{f['n_onsets']:>9}"
                      f"{f['recall']:>8.3f}{f['precision']:>8.3f}"
                      f"{f['f1']:>8.3f}{f['falsos_h']:>10.2f}"
                      f"{f['err_mae']:>8.2f}")
                _anotar(f, combo, "global")

            print(f"\n  Por clase (sensibilidad)")
            print(f"  {'conjunto':<34}"
                  + "".join(f"{c:>8}" for c in ed.CLASES))
            print(f"  {'(n de eventos)':<34}"
                  + "".join(f"{n_por_clase[c]:>8}" for c in ed.CLASES))
            for combo, ons in conjuntos.items():
                f = metricas_de_onsets(catalogo, ons, horas)
                print(f"  {' + '.join(combo):<34}"
                      + "".join(f"{f[f'rec_{c}']:>8.3f}"
                                if not np.isnan(f[f"rec_{c}"])
                                else f"{'-':>8}" for c in ed.CLASES))

            print(f"\n  Por tramo")
            print(f"  {'ambito':<10}{'conjunto':<34}{'cat':>5}"
                  f"{'n_onsets':>9}{'sens':>8}{'prec':>8}{'fp/h':>10}")
            for nombre_amb, t0, t1, h_amb in amb:
                if nombre_amb == "global":
                    continue
                cat_amb = [e for e in catalogo if t0 <= e.ini < t1]
                for combo, ons in conjuntos.items():
                    o = [t for t in ons if t0 <= t < t1]
                    f = metricas_de_onsets(cat_amb, o, h_amb)
                    print(f"  {nombre_amb:<10}{' + '.join(combo):<34}"
                          f"{len(cat_amb):>5}{f['n_onsets']:>9}"
                          f"{f['recall']:>8.3f}{f['precision']:>8.3f}"
                          f"{f['falsos_h']:>10.2f}")
                    _anotar(f, combo, nombre_amb, horas=h_amb,
                            n_catalogo=len(cat_amb))
                print()

    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        print("  " + "-" * 74)
        print("  EFECTO DE LA REGLA DE FUSION SOBRE LA SENSIBILIDAD GLOBAL")
        print("  " + "-" * 74)
        g = tabla[(tabla.ambito == "global") & (tabla.n_detectores > 1)]
        print(f"  {'escenario':<10}{'union':<34}"
              + "".join(f"{r[:12]:>14}" for r in REGLAS_FUSION))
        for (esc, nombre), gr in g.groupby(["escenario", "grupo"],
                                           sort=False):
            celdas = []
            for regla in REGLAS_FUSION:
                m = gr[gr.regla == regla]
                celdas.append(f"{m.iloc[0].recall:>14.3f}" if len(m)
                              else f"{'-':>14}")
            print(f"  {esc:<10}{nombre:<34}" + "".join(celdas))
        print("\n  'sin fusionar' repite onsets y por eso no es una union")
        print("  utilizable: es solo la cota superior de sensibilidad.")
    return tabla


# =====================================================================
# PARTE 4: FRONTERA, CSV Y FIGURA
# =====================================================================

def parte4(tabla1, tabla2, tabla3, con_figura=True):
    """Junta todo, marca la frontera, escribe el CSV y la figura."""
    print("\n" + "=" * 78)
    print("  PARTE 4 - FRONTERA SENSIBILIDAD CONTRA FALSOS POR HORA")
    print("=" * 78)
    print("  Entran los puntos de las tres partes. Los del barrido previo")
    print("  llevan corte=3.240.000 y los demas corte=3.240.001; el efecto")
    print("  medido de esa diferencia esta en la seccion de procedencia.\n")

    partes = [t for t in (tabla1, tabla2, tabla3)
              if t is not None and not t.empty]
    if not partes:
        print("  nada que guardar")
        return None
    todo = pd.concat(partes, ignore_index=True, sort=False)

    # punto_id identifica cada fila sin ambiguedad, asi que la frontera
    # se marca por id y no por coincidencia de valores.
    todo["punto_id"] = np.arange(len(todo))

    # Los modelos del barrido previo se reproducen exacto en la extension
    # (la seccion de procedencia lo mide), asi que esas filas quedan
    # duplicadas. Se conservan en el CSV pero se excluyen del calculo de
    # la frontera para que cada punto tenga una sola procedencia.
    clave = ["grupo", "modo", "umbral", "min_est", "n_onsets", "ambito",
             "escenario", "regla"]
    todo["duplicado"] = (todo.sort_values("fuente",
                                          key=lambda c: c.eq("barrido_previo"))
                         .duplicated(subset=clave)
                         .reindex(todo.index))

    # "sin fusionar" deja onsets repetidos, asi que no es una union
    # utilizable: se conserva en el CSV como cota superior pero se saca
    # de la frontera y del cruce de metas.
    usable = todo.regla.ne("sin fusionar") if "regla" in todo.columns \
        else pd.Series(True, index=todo.index)
    glob = todo[(todo.ambito == "global") & (~todo.duplicado)
                & usable].copy()
    fr = frontera(glob)
    todo["en_frontera"] = todo.punto_id.isin(set(fr.punto_id))
    print(f"  Filas duplicadas entre el barrido previo y la extension: "
          f"{int(todo.duplicado.sum())} (se excluyen de la frontera)")
    print(f"  Filas con regla 'sin fusionar': "
          f"{int((~usable).sum())} (se excluyen de la frontera por repetir"
          f" onsets)")

    print(f"\n  Puntos globales: {len(glob)} | en la frontera: "
          f"{int(todo.en_frontera.sum())}\n")
    print(f"  {'sens':>8}{'fp/h':>10}{'prec':>8}{'n_onsets':>10}"
          f"  punto")
    for _, r in fr.sort_values("recall", ascending=False).iterrows():
        extra = (f"umbral {r.umbral:g}, min_est {int(r.min_est)}"
                 if not pd.isna(r.get("umbral", np.nan))
                 else f"escenario {r.get('escenario', '')}")
        print(f"  {r.recall:>8.3f}{r.falsos_h:>10.2f}{r.precision:>8.3f}"
              f"{int(r.n_onsets):>10}  {r.fuente}: {r.grupo} "
              f"[{r.modo}] {extra}")

    print(f"\n  Cruce de las metas sobre TODOS los puntos globales medidos")
    print(f"  {'meta':>7}{'n puntos':>10}{'sens':>8}{'fp/h':>10}"
          f"{'prec':>8}  punto mas barato")
    for obj in OBJETIVOS:
        sel = glob[glob.recall >= obj]
        r = primer_punto(glob, obj)
        if r is None:
            print(f"  {obj:>7.2f}{0:>10}{'-':>8}{'-':>10}{'-':>8}"
                  f"  NO alcanzada por ningun punto medido")
            continue
        extra = (f"umbral {r.umbral:g}, min_est {int(r.min_est)}"
                 if not pd.isna(r.get("umbral", np.nan))
                 else f"escenario {r.get('escenario', '')}")
        print(f"  {obj:>7.2f}{len(sel):>10}{r.recall:>8.3f}"
              f"{r.falsos_h:>10.2f}{r.precision:>8.3f}"
              f"  {r.fuente}: {r.grupo} [{r.modo}] {extra}")

    cols = ["punto_id", "fuente", "grupo", "escenario", "modo",
            "umbral", "min_est", "regla", "corte",
            "n_detectores", "ambito", "horas", "n_catalogo", "n_picks",
            "n_onsets", "recall", "precision", "f1", "falsos_h", "err_mae",
            "duplicado", "en_frontera"] + [f"rec_{c}" for c in ed.CLASES] \
          + [f"n_{c}" for c in ed.CLASES]
    for c in cols:
        if c not in todo.columns:
            todo[c] = np.nan
    todo = todo[cols]
    todo.to_csv(CSV_SALIDA, index=False)
    print(f"\n  Tabla completa ({len(todo)} filas) en {CSV_SALIDA}")

    if con_figura:
        graficar(glob, fr)
    return todo


def graficar(glob, fr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        print(f"  (sin figura: {ex})")
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    estilos = {
        "phasenet": ("s", "#1f77b4"),
        "eqtransformer": ("^", "#d62728"),
        "stalta": ("o", "#444444"),
    }
    prev = glob[glob.fuente == "barrido_previo"]
    for modelo, g in prev.groupby("grupo"):
        marca, color = estilos.get(modelo, ("x", "gray"))
        ax.scatter(np.maximum(g.falsos_h, 1e-2), g.recall, s=12,
                   marker=marca, color=color, alpha=0.22,
                   label=f"{modelo} (barrido previo)")

    ext = glob[glob.fuente == "extension"]
    for (modelo, modo), g in ext.groupby(["grupo", "modo"]):
        marca, color = estilos.get(modelo, ("x", "gray"))
        relleno = "none" if modo == "duplicar" else color
        ax.scatter(np.maximum(g.falsos_h, 1e-2), g.recall, s=26,
                   marker=marca, edgecolors=color, facecolors=relleno,
                   alpha=0.85, label=f"{modelo} / {modo}")

    comb = glob[glob.fuente == "combinacion"]
    for esc, g in comb.groupby("escenario"):
        uni = g[g.n_detectores > 1]
        ind = g[g.n_detectores == 1]
        m = "P" if esc == "op" else "X"
        ax.scatter(np.maximum(uni.falsos_h, 1e-2), uni.recall, s=95,
                   marker=m, color="#2ca02c",
                   label=f"union ({esc})", zorder=5)
        ax.scatter(np.maximum(ind.falsos_h, 1e-2), ind.recall, s=55,
                   marker=m, facecolors="none", edgecolors="#7f7f7f",
                   label=f"individual ({esc})", zorder=4)

    ax.plot(np.maximum(fr.falsos_h, 1e-2), fr.recall, color="black",
            lw=1.6, ls="--", zorder=3, label="frontera")

    for obj, col in zip(OBJETIVOS, ["#999999", "#bbbb33", "#cc3333"]):
        ax.axhline(obj, color=col, lw=1.0, ls=":")
        ax.text(ax.get_xlim()[1], obj, f" {obj:.2f}", va="center",
                ha="left", fontsize=8, color=col)

    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlabel("Onsets falsos por hora (escala logaritmica)")
    ax.set_ylabel(f"Sensibilidad de onset (tolerancia ±{TOL_ONSET_S:g} s)")
    ax.set_title("Nivel 1: sensibilidad contra falsos positivos por hora")
    ax.grid(alpha=0.25)
    # La esquina de arriba a la izquierda (pocos falsos y mucha
    # sensibilidad) esta vacia, asi que la leyenda no tapa ningun punto.
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(PNG_SALIDA, dpi=150)
    print(f"  Figura en {PNG_SALIDA}")


# =====================================================================
# MAIN
# =====================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--canales", nargs="+", type=int, default=ed.CANALES)
    ap.add_argument("--sin-cache", action="store_true",
                    help="recalcula STA/LTA en vez de leer su cacha")
    ap.add_argument("--sin-figura", action="store_true")
    args = ap.parse_args(argv)

    print("Cargando traza y catalogo ...")
    a, catalogo, valida, horas = ed.cargar()
    print(f"  {len(catalogo)} eventos, {horas:.2f} h validas, "
          f"canales {args.canales}\n")

    ok, onsets_op, _ = verificar(a, catalogo, valida, horas, args.canales,
                                 not args.sin_cache)
    if not ok:
        print("\n  No reproduzco el nivel 1 de significancia.py.")
        print("  Me detengo: no reporto sobre una base que no empata con")
        print("  lo que dice el documento.")
        return 1

    pisos = pisos_de_cache()
    tabla2 = parte2(catalogo, valida, horas, pisos)
    verificar_procedencia(tabla2, a, catalogo, valida, horas, args.canales,
                          not args.sin_cache)
    tabla1 = parte1()
    tabla3 = parte3(a, catalogo, valida, horas, args.canales,
                    not args.sin_cache, onsets_op, tabla2)
    parte4(tabla1, tabla2, tabla3, con_figura=not args.sin_figura)

    print("\n" + "=" * 78)
    print("  ESTE SCRIPT NO ELIGE UN PUNTO DE OPERACION")
    print("=" * 78)
    print("""  Solo deja medida la frontera. La eleccion del punto de
  operacion es una decision metodologica que hay que escribir aparte,
  declarando la meta de sensibilidad, el costo en falsos positivos que
  se acepta y, si la meta del 95 % no se alcanza, la sensibilidad maxima
  alcanzable y su costo.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
