#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lista de ventanas del conjunto de entrenamiento (NVCHVC_8192.zip) que se excluyen por estar
o parecerse a eventos de la traza continua. Lee la busqueda completa de
superposicion_pool_traza.py --todo y escribe config/exclusiones_pool.csv.

Uso: python lista_exclusiones.py
"""

import os
import sys

import numpy as np
import pandas as pd

DIR_TESIS = os.path.expanduser("~/tesis")
CSV_BUSQUEDA = os.path.join(DIR_TESIS, "resultados", "superposicion_pool_traza.csv")
CSV_NULO = os.path.join(DIR_TESIS, "resultados", "superposicion_pool_traza_nulo.csv")
CSV_SALIDA = os.path.join(DIR_TESIS, "config", "exclusiones_pool.csv")
N_POOL = 9102

# regla de exclusion: ambos umbrales quedan sobre el maximo del grupo de control (nulo)
UMBRAL_MEDIA = 0.90
UMBRAL_1EST = 0.96
# los motivos se fijaron despues de ver los datos; solo describen, no cambian que se excluye
COPIA_MEDIA = 0.93
COPIA_1EST_AV = 0.98
ESPERADO = {"copia": 27, "parecida": 8, "dudosa": 7}


def es_si(v):
    return str(v).strip().lower() == "true"


def main():
    for ruta in (CSV_BUSQUEDA, CSV_NULO):
        if not os.path.exists(ruta):
            raise SystemExit(f"Falta {ruta}: correr antes superposicion_pool_traza.py --todo")
    df = pd.read_csv(CSV_BUSQUEDA)
    nulo = pd.read_csv(CSV_NULO)
    print(f"Busqueda: {len(df):,} ventanas de {N_POOL:,}; conjuntos: {df.conjunto.value_counts().to_dict()}")
    if len(df) != N_POOL or df.archivo.duplicated().any():
        raise SystemExit("DETENIDO: la busqueda no cubre las 9.102 ventanas una sola vez (falta --todo).")

    # control: el maximo del nulo tiene que quedar bajo los umbrales
    max_media = float(np.nanmax(nulo[["ncc_ventana", "ncc_evento"]].to_numpy()))
    limpio = nulo[~nulo.av_sin_crudo_1est.map(es_si)]
    max_1est = float(limpio.ncc1_evento.max())
    print(f"Grupo de control ({len(nulo)} ventanas de fechas ajenas a la traza): "
          f"NCC media max {max_media:.4f} < {UMBRAL_MEDIA}; "
          f"estacion sola max {max_1est:.4f} < {UMBRAL_1EST} (sin los AV sin crudo)")
    if not (max_media < UMBRAL_MEDIA and max_1est < UMBRAL_1EST):
        raise SystemExit("DETENIDO: algun umbral no queda sobre el maximo del control.")

    media = df[["ncc_ventana", "ncc_evento"]].max(axis=1)
    sola = df.ncc1_evento
    excl = (media >= UMBRAL_MEDIA) | (sola >= UMBRAL_1EST)
    x = df[excl].copy()
    x["ncc_media"] = media[excl]
    x["ncc_estacion_sola"] = sola[excl]
    av = x.av_sin_crudo_1est.map(es_si)
    x["motivo"] = np.where((x.ncc_media >= COPIA_MEDIA) | ((x.ncc_estacion_sola >= COPIA_1EST_AV) & av),
                           "copia", np.where(x.ncc_media >= UMBRAL_MEDIA, "parecida", "dudosa"))

    # posicion en la traza: la del modo que da la coincidencia
    usa_media = x.ncc_media >= UMBRAL_MEDIA
    modo = np.where(usa_media, np.where(x.ncc_evento.fillna(-1) >= x.ncc_ventana.fillna(-1), "evento", "ventana"), "1est")
    x["modo"] = modo
    for col in ("tramo", "utc", "cat_tipo", "cat_iou", "cat_idx"):
        x[f"{col}_traza"] = [getattr(r, f"{col}_{m}") for r, m in zip(x.itertuples(), modo)]
    obs = []
    for r, m in zip(x.itertuples(), modo):
        partes = []
        if str(r.cat_tipo_traza) not in ("", "nan") and r.cat_tipo_traza != r.clase:
            partes.append(f"clase del pool distinta de la del evento de la traza ({r.cat_tipo_traza})")
        if es_si(getattr(r, f"av_sin_crudo_{m}")):
            partes.append("cae sobre un AV del tramo 1 sin senal en el crudo de FRE")
        obs.append("; ".join(partes))
    x["observacion"] = obs

    n = x.motivo.value_counts().to_dict()
    print(f"\nExcluidas: {len(x)} ({len(x) / N_POOL:.2%} del conjunto); por motivo: {n}; "
          f"por clase: {x.clase.value_counts().to_dict()}")
    cols = ["archivo", "clase", "motivo", "modo", "ncc_media", "ncc_estacion_sola", "tramo_traza",
            "utc_traza", "cat_tipo_traza", "cat_iou_traza", "observacion"]
    x = x.sort_values(["motivo", "clase", "archivo"])
    print(x[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if {k: n.get(k, 0) for k in ESPERADO} != ESPERADO:
        print(f"\nDETENIDO: se esperaba {ESPERADO}; no se escribe la lista. Revisar la diferencia.")
        sys.exit(1)
    os.makedirs(os.path.dirname(CSV_SALIDA), exist_ok=True)
    x[cols].to_csv(CSV_SALIDA, index=False, float_format="%.4f")
    print(f"\nLista escrita en {CSV_SALIDA}")


if __name__ == "__main__":
    main()
