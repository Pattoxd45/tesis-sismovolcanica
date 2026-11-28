#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deteccion y recorte en las ventanas de test de Jose, con la misma configuracion y la misma forma de
contar que recorte_ventanas_jose.py (PhaseNet P y S, recorte por energia con corte, ruido solo dentro
de la ventana, que es como se hicieron los recortes del clasificador): por clase, cajas detectadas e
IoU del recorte; recortes sin caja, falsas alarmas por hora y FDR; tiempo por ventana.

Uso: python deteccion_test.py [--limite 50]
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import recorte_ventanas_jose as rv

lib, ej, ep = rv.lib, rv.ej, rv.ep
SPLIT = "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N ventanas, para probar")
    args = ap.parse_args()

    with open(os.path.join(lib.DIR_SALIDA, "configuracion_elegida.json")) as f:
        cfg = json.load(f)[rv.FLUJO]
    up, us = cfg["P_threshold"], cfg["S_threshold"]
    kw = dict(overlap=int(cfg["overlap"]), blinding=(int(cfg["blinding"]), int(cfg["blinding"])),
              stacking=cfg["stacking"])
    reg = pd.read_csv(ej.REGISTRO).set_index("evento")
    ev = [e for e in ep.completas(ej.REGISTRO) if reg.loc[e, "split"] == SPLIT]
    if args.limite:
        ev = ev[:args.limite]
    cajas = rv.cargar_cajas()
    print(f"{len(ev)} ventanas de {SPLIT} completas; PhaseNet P_threshold {up}, S_threshold {us}, overlap "
          f"{kw['overlap']}, blinding {kw['blinding'][0]}, {kw['stacking']}; ruido solo dentro de la ventana")

    m = rv.an.cargar_modelo("phasenet")
    sos = lib.sos_filtro(lib.FILTROS[rv.FILTRO])
    filas_caja, filas_ven, t_ini = [], [], time.time()
    for k, e in enumerate(ev, 1):
        t0 = time.time()
        w = rv.np.load(os.path.join(ej.DIR_SALIDA, f"{e}_zne.npz"))
        pi, pf, _, ivs = rv.procesar({c: w[c] for c in ("Z", "N", "E")}, 0, m, kw, up, us, sos)
        det, iou, sin_caja = rv.evaluar(cajas[e], pi, pf, ivs, ej.N_VENTANA)
        filas_caja += [dict(ventana=e, tipo=r.tipo_marca, clase=r.clase, ini_s=r.inicio_rel_s, fin_s=r.fin_rel_s,
                            detectada=bool(d), iou=float(v)) for r, d, v in zip(cajas[e].itertuples(), det, iou)]
        filas_ven.append(dict(ventana=e, recortes=len(ivs), sin_caja=sin_caja, segundos=time.time() - t0))
        if k % 200 == 0:
            print(f"  {k}/{len(ev)} ventanas en {(time.time() - t_ini) / 60:.1f} min", flush=True)
    t, v = pd.DataFrame(filas_caja), pd.DataFrame(filas_ven)
    print(f"\nTiempo: {(time.time() - t_ini) / 60:.1f} min ({v.segundos.mean():.2f} s por ventana de "
          f"{ej.N_VENTANA / lib.FS:.0f} s)")
    horas = len(v) * ej.N_VENTANA / lib.FS / 3600
    print(f"{int(v.recortes.sum())} recortes ({v.recortes.mean():.2f} por ventana); {int(v.sin_caja.sum())} sin caja: "
          f"{v.sin_caja.sum() / horas:.1f} falsas alarmas por hora, FDR {v.sin_caja.sum() / max(v.recortes.sum(), 1):.3f}")
    rv.linea(t[t.tipo == "PRINCIPAL"], "evento principal de cada ventana")
    rv.linea(t, "todas las cajas (principales y extra)")
    os.makedirs(rv.DIR_SALIDA, exist_ok=True)
    t.to_csv(os.path.join(rv.DIR_SALIDA, f"deteccion_{SPLIT}_cajas.csv"), index=False)
    v.to_csv(os.path.join(rv.DIR_SALIDA, f"deteccion_{SPLIT}_ventanas.csv"), index=False)
    print(f"\nTablas en {rv.DIR_SALIDA}")


if __name__ == "__main__":
    main()
