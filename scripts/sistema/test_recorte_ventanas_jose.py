#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para extraer_previo_jose.py y recorte_ventanas_jose.py: periodo de ruido con y sin
los 120 s previos, recortes llevados a la ventana y comparacion con las cajas.

Uso: python test_recorte_ventanas_jose.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import recorte_ventanas_jose as rv
import extraer_previo_jose as ep

FS = rv.FS
N = 30000
# P_threshold y S_threshold de la configuracion elegida de PhaseNet
UP, US = 0.40, 0.05


class Falso:
    """Curva de P con un triangulo de 0,9 y 1 s de ancho en la muestra dada, contada desde el inicio de la ventana."""
    def __init__(self, pico):
        self.pico = pico

    def annotate(self, st, **kw):
        a = int(round(st[0].stats.starttime.timestamp * FS))
        c = np.zeros(st[0].stats.npts)
        i = np.arange(c.size)
        c[:] = np.clip(0.9 * (1 - np.abs(i - self.pico) / 50), 0, None)
        stats = SimpleNamespace(channel="PhaseNet_P", sampling_rate=FS, starttime=SimpleNamespace(timestamp=a / FS))
        return [SimpleNamespace(stats=stats, data=c, id="P")]


def senal(n_previo, ini_ev, fin_ev, semilla=0):
    rng = np.random.default_rng(semilla)
    x = {c: rng.normal(0, 1, n_previo + N) for c in ("Z", "N", "E")}
    for c in x:
        x[c][n_previo + ini_ev:n_previo + fin_ev] *= 30
    return x


def caso(nombre, ok):
    print(f"  {'OK   ' if ok else 'FALLA'} {nombre}")
    return ok


def main():
    todo = True
    sos = rv.lib.sos_filtro(rv.lib.FILTROS["f1_15"])
    kw = {}

    x = senal(ep.N_PREVIO, 300, 1300)
    sin = rv.procesar({c: v[ep.N_PREVIO:] for c, v in x.items()}, 0, Falso(300), kw, UP, US, sos)
    con = rv.procesar(x, ep.N_PREVIO, Falso(300), kw, UP, US, sos)
    todo &= caso("evento a los 3 s: sin los 120 s previos no hay ruido y no hay recorte", sin[3] == [])
    todo &= caso("evento a los 3 s: con los 120 s previos el recorte empieza en la ventana y termina tras el evento",
                 len(con[3]) == 1 and con[3][0][0] == 50 and 1300 <= con[3][0][1] <= 1900
                 and con[2].tolist() == [300])

    x = senal(ep.N_PREVIO, 15000, 16000, semilla=1)
    sin = rv.procesar({c: v[ep.N_PREVIO:] for c, v in x.items()}, 0, Falso(15000), kw, UP, US, sos)
    con = rv.procesar(x, ep.N_PREVIO, Falso(15000), kw, UP, US, sos)
    todo &= caso("evento a los 150 s: las dos formas dan el mismo inicio y casi el mismo fin",
                 len(sin[3]) == 1 and len(con[3]) == 1 and sin[3][0][0] == con[3][0][0] == 14750
                 and abs(sin[3][0][1] - con[3][0][1]) <= 100)
    todo &= caso("los intervalos de P quedan en muestras de la ventana",
                 sin[0].tolist() == con[0].tolist() and 14950 <= sin[0][0] <= 15000)

    cajas = pd.DataFrame(dict(ini=[15000, 25000], fin=[17000, 26000], clase=["VT", "LP"]))
    det, iou, sin_caja = rv.evaluar(cajas, np.array([15100]), np.array([15200]), [(14800, 17200), (2000, 2500)], N)
    todo &= caso("caja cruzada por un intervalo y con su recorte; otra sin nada; un recorte sin caja",
                 det.tolist() == [True, False] and abs(iou[0] - 2000 / 2400) < 1e-9 and iou[1] == 0 and sin_caja == 1)

    ruta = os.path.join(tempfile.mkdtemp(), "reg.csv")
    pd.DataFrame(dict(evento=["a", "b", "c", "d"], con_dato_Z=[1, 1, 1, 0.99], con_dato_N=[1, 0, 1, 1],
                      con_dato_E=[1, 1, 1, 1], tablas_sin_leer=[np.nan, np.nan, "fre_n_2019_04_08", np.nan])
                 ).to_csv(ruta, index=False)
    todo &= caso("ventanas completas: Z, N y E al 100 % y sin tablas cortas", ep.completas(ruta) == ["a"])

    print("TODOS PASAN" if todo else "HAY FALLAS")
    sys.exit(0 if todo else 1)


if __name__ == "__main__":
    main()
