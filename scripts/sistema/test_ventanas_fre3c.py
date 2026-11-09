#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para ventanas_fre3c.py: avisos de cada ventana en muestras de la traza completa y
union de los avisos repetidos entre ventanas.

Uso: python test_ventanas_fre3c.py
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import ventanas_fre3c as vf

FS = vf.FS
UP, US = 0.40, 0.05


def traza(canal, inicio, curva):
    st = SimpleNamespace(channel=canal, sampling_rate=FS, starttime=SimpleNamespace(timestamp=inicio / FS))
    return SimpleNamespace(stats=st, data=np.asarray(curva, dtype=float), id=canal)


def curva(n, picos):
    """Curva de largo n con un triangulo de altura 0,9 y 1 s de ancho en cada pico (en muestras locales)."""
    c = np.zeros(n)
    for p in picos:
        a, b = max(p - 50, 0), min(p + 50, n)
        c[a:b] = np.maximum(c[a:b], 0.9 * (1 - np.abs(np.arange(a, b) - p) / 50))
    return c


def caso(nombre, ok):
    print(f"  {'OK   ' if ok else 'FALLA'} {nombre}")
    return ok


def main():
    todo = True
    n = 3000

    v1 = vf.avisos_de_ventana([traza("PhaseNet_P", 0, curva(n, [1000]))], UP, US)
    v2 = vf.avisos_de_ventana([traza("PhaseNet_P", 500, curva(n, [500]))], UP, US)
    todo &= caso("el aviso queda en muestras de la traza completa",
                 len(v1) == 1 and len(v2) == 1 and v1[0]["pico"] == 1000 and v2[0]["pico"] == 1000)

    pi, pf, av = vf.juntar(pd.DataFrame(v1 + v2))
    todo &= caso("el mismo aviso en dos ventanas cuenta una vez", pi.size == 1 and av.tolist() == [1000])

    ruido = vf.avisos_de_ventana([traza("PhaseNet_N", 0, curva(n, [1000]))], UP, US)
    todo &= caso("la curva de ruido no da avisos", ruido == [])

    ps = vf.avisos_de_ventana([traza("PhaseNet_P", 0, curva(n, [1000])), traza("PhaseNet_S", 0, curva(n, [1300]))], UP, US)
    pi, pf, av = vf.juntar(pd.DataFrame(ps))
    todo &= caso("P y S a 3 s son un aviso, con la hora de la P", pi.size == 1 and av.tolist() == [1000])

    lejos = vf.avisos_de_ventana([traza("PhaseNet_P", 0, curva(n, [1000, 2000]))], UP, US)
    pi, pf, av = vf.juntar(pd.DataFrame(lejos))
    todo &= caso("dos avisos a 10 s son dos", pi.size == 2 and av.tolist() == [1000, 2000])

    s_baja = vf.avisos_de_ventana([traza("PhaseNet_S", 0, 0.1 * curva(n, [1000]))], UP, US)
    p_baja = vf.avisos_de_ventana([traza("PhaseNet_P", 0, 0.1 * curva(n, [1000]))], UP, US)
    todo &= caso("cada onda usa su umbral (S 0,05 avisa, P 0,40 no)", len(s_baja) == 1 and p_baja == [])

    pi, pf, av = vf.juntar(pd.DataFrame())
    todo &= caso("sin avisos no falla", pi.size == 0 and av.size == 0)

    print("TODOS PASAN" if todo else "HAY FALLAS")
    sys.exit(0 if todo else 1)


if __name__ == "__main__":
    main()
