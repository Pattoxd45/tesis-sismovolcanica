#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumen de la extraccion de Z, N y E para las ventanas de Jose: si las ventanas con la Z distinta a
la de Jose son las que el saco de otra estacion (segun su metadata_dataset.csv) y cuantas ventanas
tienen las tres componentes completas, por particion y clase.

Uso: python resumen_extraccion.py
"""

import os
import sys

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import extraer_ventanas_jose as ej

META_JOSE = os.path.join(ej.DIR_JOSE, "dataset_SOLO_Z_300s", "metadata_dataset.csv")
SALIDA = os.path.expanduser("~/tesis/resultados/sistema/ventanas_incompletas.csv")


def estado(r):
    if isinstance(r.tablas_sin_leer, str) and r.tablas_sin_leer:
        return "tabla corta sin leer"
    if r.con_dato_Z == 1 and r.con_dato_N == 1 and r.con_dato_E == 1:
        return "completa"
    if r.con_dato_N == 0 or r.con_dato_E == 0:
        return "sin N o sin E"
    return "con huecos"


def main():
    d = pd.read_csv(ej.REGISTRO)
    m = pd.read_csv(META_JOSE, encoding="utf-8-sig")
    m["estacion_jose"] = m.base.str.lower().str.extract(r"w_(\w+?)z\$hh", expand=False).str.upper()
    d = d.merge(m[["evento", "estacion_jose"]], on="evento", how="left")
    d["z"] = d.z_igual_jose.map(lambda f: "sin comparar" if pd.isna(f) else "igual" if f == 1 else "distinta")
    print(f"{len(d)} ventanas extraidas ({d.evento.nunique()} distintas)")

    print("\n1. Z PROPIA (FRE) CONTRA LA DE JOSE, SEGUN LA ESTACION DE LA QUE EL LA SACO")
    print(pd.crosstab(d.estacion_jose.fillna("sin dato"), d.z, margins=True, margins_name="total").to_string())
    raras = d[((d.z == "distinta") & (d.estacion_jose == "FRE")) | (d.z == "sin comparar")]
    for r in raras.itertuples():
        npy = ej.z_de_jose(r.evento, r.clase)
        print(f"   {r.evento} {r.split}: Z {r.z}; estacion de Jose {r.estacion_jose}; Z propia con dato {r.con_dato_Z:.1%}; "
              f"npy de Jose: {'no esta' if npy is None else f'{npy.size} muestras'}")

    print("\n2. COMPONENTES POR VENTANA")
    d["estado"] = d.apply(estado, axis=1)
    print(pd.crosstab(d.estado, d.split, margins=True, margins_name="total").to_string())
    print("\n   test por clase:")
    t = d[d.split == "test"]
    print(pd.crosstab(t.estado, t.clase, margins=True, margins_name="total").to_string())
    h = d[d.estado == "con huecos"]
    if len(h):
        print(f"\n   con huecos: dato minimo en la ventana, mediana {h[['con_dato_Z', 'con_dato_N', 'con_dato_E']].min(axis=1).median():.1%}")

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    cols = ["evento", "clase", "split", "estado", "con_dato_Z", "con_dato_N", "con_dato_E", "tablas_sin_leer",
            "z", "estacion_jose"]
    d[d.estado != "completa"][cols].to_csv(SALIDA, index=False)
    print(f"\nVentanas que no estan completas en {SALIDA}")


if __name__ == "__main__":
    main()
