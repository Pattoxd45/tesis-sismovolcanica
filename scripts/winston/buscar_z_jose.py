#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Para las ventanas extraidas cuya Z no es igual a la de Jose: busca la Z de Jose en la senal de FRE
desde una hora antes hasta una hora despues de la ventana (solo con las tablas ya cargadas en la
MariaDB local; no se conecta al NAS) y dice con que desfase aparece.

Uso: python buscar_z_jose.py [--margen 3600]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import extraer_winston as ew
import extraer_ventanas_jose as ej
import winston_lib as wl

FS = wl.FS
# muestras del trozo de la Z de Jose que se busca: 3 s, suficiente para que no calce por azar con enteros
TROZO = 300


def buscar(largo, j, k0):
    """Desfases (muestras, respecto del inicio de la ventana) en que un trozo con dato de j aparece en largo."""
    for a in range(0, j.size - TROZO + 1, TROZO):
        p = j[a:a + TROZO]
        if np.isfinite(p).all() and np.ptp(p) > 0:
            break
    else:
        return []
    cand = np.flatnonzero(largo == p[0])
    return [int(c - a - k0) for c in cand if c + TROZO <= largo.size and np.array_equal(largo[c:c + TROZO], p)]


def igual_en(largo, j, k0, k):
    """Parte de las muestras con dato en ambas iguales, con la Z de Jose desplazada k muestras."""
    s = largo[k0 + k:k0 + k + j.size]
    if s.size < j.size:
        return np.nan
    m = np.isfinite(s) & np.isfinite(j)
    return float(np.mean(s[m] == j[m])) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margen", type=float, default=3600.0, help="segundos antes y despues de la ventana")
    args = ap.parse_args()

    d = pd.read_csv(ej.REGISTRO)
    cob = pd.read_csv(ej.COBERTURA).set_index("evento_principal")
    malas = d[d.z_igual_jose < 1]
    cargadas = ew.tablas_cargadas()
    k0 = int(round(args.margen * FS))
    print(f"{len(malas)} ventanas con la Z distinta a la de Jose; se busca su Z desde {args.margen:.0f} s antes "
          f"hasta {args.margen:.0f} s despues")
    for r in malas.itertuples():
        w = cob.loc[r.evento]
        dias = ew.dias(ej.utc(w.inicio_ventana_w - args.margen), ej.utc(w.fin_ventana_w + args.margen))
        faltan = [x for x in dias if ew.tabla_local(ej.ESTACION, "Z", x) not in cargadas]
        t0 = w.inicio_ventana_w + wl.J2K_A_UNIX - args.margen
        senal, _ = ew.senal_entre(ej.ESTACION, dias, t0, ej.N_VENTANA + 2 * k0, cargadas)
        z = senal["Z"]
        j = ej.z_de_jose(r.evento, r.clase)
        hits = buscar(z, j, k0)
        texto = ", ".join(f"{k / FS:+.2f} s (igual {igual_en(z, j, k0, k):.1%})" for k in hits[:5]) or "no aparece"
        print(f"  {r.evento} {r.split}: ventana desde {ej.utc(w.inicio_ventana_w):%Y-%m-%d %H:%M:%S} UTC; "
              f"Z propia con dato {r.con_dato_Z:.1%}, Z de Jose con dato {np.isfinite(j).mean():.1%}; "
              f"la Z de Jose aparece en: {texto}"
              + (f"; dias sin tabla cargada: {', '.join(map(str, faltan))}" if faltan else ""))


if __name__ == "__main__":
    main()
