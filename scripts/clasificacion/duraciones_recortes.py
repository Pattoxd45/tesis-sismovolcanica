#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duracion de los recortes (por etiqueta) y de los eventos del catalogo (cajas de Jose, por clase), para
elegir una duracion minima de los recortes y la duracion fija de la segunda forma de armar las
imagenes: cuantos recortes quedarian fuera con cada minimo y cuantos se cortarian con cada duracion.

Uso: python duraciones_recortes.py
"""

import os

import pandas as pd

DIR_TESIS = os.path.expanduser("~/tesis")
INDICE = os.path.join(DIR_TESIS, "resultados", "sistema", "espectrogramas_recortes.csv")
CAJAS = os.path.join(DIR_TESIS, "datos", "jose", "Pato", "metadata_dataset", "metadata",
                     "bounding_boxes_para_entrenamiento.csv")
ORDEN = ["VT", "LP", "TR", "AV", "IC", "sin caja", "fuera"]
# duraciones (s) en que se mide; son puntos de comparacion, no parametros del sistema
MINIMAS = [2, 3, 4, 5, 6, 8, 10]
FIJAS = [20, 30, 40, 60, 90, 120]
# una caja que empieza en 0 o termina en 300 s esta cortada por el borde de la ventana
VENTANA_S = 300.0


def percentiles(dur, grupo):
    q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    t = dur.groupby(grupo).quantile(q).unstack().round(1)
    t.columns = [f"p{int(x * 100)}" for x in q]
    t["max"] = dur.groupby(grupo).max().round(1)
    t.insert(0, "n", dur.groupby(grupo).size())
    return t.reindex([k for k in ORDEN if k in t.index])


def fraccion(dur, grupo, cortes, mayor):
    """Por grupo y corte: parte de los recortes con duracion mayor o igual (mayor=True) o menor al corte."""
    filas = {}
    for c in cortes:
        x = (dur >= c) if mayor else (dur < c)
        filas[f"{c} s"] = x.groupby(grupo).mean()
    t = pd.DataFrame(filas).round(3)
    return t.reindex([k for k in ORDEN if k in t.index])


def main():
    idx = pd.read_csv(INDICE)
    print(f"{len(idx)} recortes de {idx.ventana.nunique()} ventanas")
    print("\n1. DURACION DE LOS RECORTES POR ETIQUETA (s)")
    print(percentiles(idx.dur_s, idx.etiqueta).to_string())

    c = pd.read_csv(CAJAS, encoding="utf-8-sig")
    c = c[c.ventana_evento_principal.isin(idx.ventana)]
    c["dur_s"] = c.fin_rel_s - c.inicio_rel_s
    borde = (c.inicio_rel_s <= 0) | (c.fin_rel_s >= VENTANA_S)
    print(f"\n2. DURACION DE LOS EVENTOS DEL CATALOGO (cajas de Jose en las mismas ventanas; {int(borde.sum())} de "
          f"{len(c)} cortadas por el borde de la ventana)")
    print(percentiles(c.dur_s, c.clase).to_string())

    print("\n3. DURACION MINIMA: parte de los recortes que quedaria fuera (mas cortos que el minimo)")
    print(fraccion(idx.dur_s, idx.etiqueta, MINIMAS, mayor=False).to_string())

    print("\n4. DURACION FIJA: parte que se cortaria (dura mas que la duracion fija)")
    print("   recortes:")
    print(fraccion(idx.dur_s, idx.etiqueta, FIJAS, mayor=True).to_string())
    print("   eventos del catalogo:")
    print(fraccion(c.dur_s, c.clase, FIJAS, mayor=True).to_string())


if __name__ == "__main__":
    main()
