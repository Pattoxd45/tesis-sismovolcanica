#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para la deteccion por interseccion de interseccion_fre3c.py.

Uso: python test_interseccion_fre3c.py
"""

import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import interseccion_fre3c as ix

FS = int(ix.FS)


def eventos(*ivs):
    return pd.DataFrame(dict(ini=[a * FS for a, _ in ivs], fin=[b * FS for _, b in ivs],
                             clase=["VT"] * len(ivs), tramo=[1] * len(ivs)))


def correr(pred, ev, valida=None, zona_extra=()):
    valida = np.ones(10_000 * FS, dtype=bool) if valida is None else valida
    zi = list(ev.ini) + [a * FS for a, _ in zona_extra]
    zf = list(ev.fin) + [b * FS for _, b in zona_extra]
    zona = ix.unir(np.asarray(zi), np.asarray(zf))
    pi = np.asarray([a * FS for a, _ in pred], dtype=np.int64)
    pf = np.asarray([b * FS for _, b in pred], dtype=np.int64)
    return ix.evaluar_interseccion(pi, pf, ev, zona, valida)


def caso(nombre, ok):
    print(f"  {'OK   ' if ok else 'FALLA'} {nombre}")
    return ok


def main():
    todo = True
    ev = eventos((100, 150))

    r = correr([(120, 121)], ev)
    todo &= caso("aviso corto dentro del evento detecta", r["asignado"][0] == 0)

    r = correr([(90, 100)], ev)
    todo &= caso("tocar el inicio en una muestra detecta", r["asignado"][0] == 0)

    r = correr([(150, 160)], ev)
    todo &= caso("tocar el fin en una muestra detecta", r["asignado"][0] == 0)

    r = correr([(80, 99)], ev)
    todo &= caso("terminar antes del inicio no detecta y es FP",
                 r["asignado"][0] == -1 and r["cat"][0] == "externa")

    r = correr([(60, 400)], ev)
    todo &= caso("un intervalo que cubre todo el evento detecta", r["asignado"][0] == 0)

    ev2 = eventos((100, 150), (160, 200))
    r = correr([(140, 170)], ev2)
    todo &= caso("un intervalo que cruza dos eventos cuenta para uno solo",
                 int((r["asignado"] >= 0).sum()) == 1)

    r = correr([(140, 170), (180, 181)], ev2)
    todo &= caso("la asignacion maximiza los eventos detectados",
                 int((r["asignado"] >= 0).sum()) == 2)

    r = correr([(110, 111), (120, 121)], ev)
    todo &= caso("segundo intervalo en el mismo evento es interna, no FP",
                 sorted(r["cat"].tolist()) == ["acierto", "interna"])

    r = correr([(500, 501)], ev, zona_extra=[(490, 510)])
    todo &= caso("intervalo dentro de un AV o IC no es FP", r["cat"][0] == "interna")

    valida = np.ones(10_000 * FS, dtype=bool)
    valida[115 * FS:125 * FS] = False
    r = correr([(120, 121)], ev, valida=valida)
    todo &= caso("intervalo que empieza sin registro se descarta", r["onsets"].size == 0)

    gi, gf = ix.agrupar_intervalos(np.array([0, 3, 9, 12]) * FS, np.array([1, 4, 10, 30]) * FS)
    todo &= caso("agrupacion anclada de P y S (5 s, sin encadenar)",
                 gi.tolist() == [0, 9 * FS] and gf.tolist() == [4 * FS, 30 * FS])

    print("TODOS PASAN" if todo else "HAY FALLAS")
    sys.exit(0 if todo else 1)


if __name__ == "__main__":
    main()
