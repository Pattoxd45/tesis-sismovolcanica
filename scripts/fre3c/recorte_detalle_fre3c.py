#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recorte por energia de PhaseNet en detalle: efecto del periodo de ruido y del percentil, y si
el recorte alcanza el inicio del evento segun la onda (P o S) que dio el aviso.

Uso: python recorte_detalle_fre3c.py [--umbral 0.10] [--filtro f1_15]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
import fre3c_lib as lib
import evaluar_detectores as ed
import recorte_fre3c as rc
import iou_fre3c as iu

FS = lib.FS
# (desde, hasta) en segundos antes del aviso; (120, 5) es el valor en uso
RUIDOS_S = [(30, 5), (60, 5), (120, 5), (240, 5), (120, 10), (120, 2.5)]
PERCENTILES = [10, 20, 50]


def medir(ivs, inc):
    pe = pd.DataFrame(iu.por_evento("x", ivs, inc))
    v = pe[pe.clase.isin(lib.VOLCANICAS)]
    tr = v[v.clase == "TR"]
    return dict(toca=int((v.iou > 0).sum()), k30=int((v.iou >= 0.3).sum()), k50=int((v.iou >= 0.5).sum()),
                iou_mediano=v[v.iou > 0].iou.median(), tr_k30=int((tr.iou >= 0.3).sum()),
                tr_iou_mediano=tr[tr.iou > 0].iou.median(), recortes=len(ivs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.10)
    ap.add_argument("--filtro", default="f1_15", choices=list(lib.FILTROS))
    args = ap.parse_args()

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    z = np.load(lib.ruta_datos(args.filtro))
    valida = z["valida"]
    env = rc.envolvente(z["Z"], z["N"], z["E"])
    pn = np.load(lib.ruta_anot("phasenet", args.filtro))
    curvas = {k: pn[k] for k in ("P", "S")}
    o = rc.onsets_pn_ps(curvas, args.umbral, valida)
    n_volc = int(inc.clase.isin(lib.VOLCANICAS).sum())

    print(f"RECORTE POR ENERGIA CON CORTE EN DETALLE (PhaseNet P y S, umbral {args.umbral:.2f}, "
          f"filtro {args.filtro}; {n_volc} volcanicos)")
    print("\n1. PERIODO DE RUIDO Y PERCENTIL (el resto del recorte sin cambios)")
    print(f"   {'ruido (s antes)':>16} {'percentil':>9} {'recortes':>8} {'toca':>5} {'0.3':>5} {'0.5':>5} "
          f"{'IoU med':>8} {'TR 0.3':>7} {'TR IoU med':>11}")
    original = (rc.RUIDO_S, rc.PERCENTIL_RUIDO)
    filas = []
    for desde, hasta in RUIDOS_S:
        for p in PERCENTILES:
            rc.RUIDO_S, rc.PERCENTIL_RUIDO = (float(desde), float(hasta)), p
            m = medir(rc.recortes(o, env, valida, rc.K_RUIDO, corte=True), inc)
            filas.append(dict(ruido_desde_s=desde, ruido_hasta_s=hasta, percentil=p, **m))
            marca = "  <- en uso" if (float(desde), float(hasta)) == original[0] and p == original[1] else ""
            print(f"   {f'{desde:g} a {hasta:g}':>16} {p:>9} {m['recortes']:>8} {m['toca']:>5} {m['k30']:>5} "
                  f"{m['k50']:>5} {m['iou_mediano']:>8.2f} {m['tr_k30']:>7} {m['tr_iou_mediano']:>11.2f}{marca}")
    rc.RUIDO_S, rc.PERCENTIL_RUIDO = original

    # con los valores en uso: que onda dio el aviso de cada recorte y donde empieza respecto del evento
    ivs = rc.recortes(o, env, valida, rc.K_RUIDO, corte=True)
    pico_p = set(lib.disparos_por_tramo(curvas["P"], args.umbral)[2].tolist())
    pico_s = set(lib.disparos_por_tramo(curvas["S"], args.umbral)[2].tolist())
    m = int(rc.MARGEN_PREVIO_S * FS)
    cat = [ed.Intervalo(int(r.ini), int(r.fin), r.clase) for r in inc.itertuples()]
    pares, _, _ = ed.emparejar(cat, [ed.Intervalo(a, b) for a, b in ivs], iu.TOCA)
    filas_i = []
    for g, d, v in pares:
        aviso = d.ini + m
        onda = "P" if aviso in pico_p else ("S" if aviso in pico_s else "?")
        filas_i.append(dict(clase=g.clase, onda=onda, iou=v, inicio_rel_s=(d.ini - g.ini) / FS,
                            aviso_rel_s=(aviso - g.ini) / FS))
    di = pd.DataFrame(filas_i)
    di = di[di.clase.isin(lib.VOLCANICAS)]
    print("\n2. INICIO DEL RECORTE SEGUN LA ONDA QUE DIO EL AVISO (volcanicos con recorte que los toca)")
    print("   '?' = el recorte empieza en el borde de un tramo y no se puede saber la onda")
    print(f"   {'onda':>4} {'clase':>5} {'eventos':>7} {'empieza antes':>13} {'empieza despues':>15} "
          f"{'aviso tras inicio, mediana (s)':>31} {'IoU med':>8}")
    for onda in ("P", "S", "?"):
        for cl in lib.VOLCANICAS + ["todas"]:
            g = di[di.onda == onda] if cl == "todas" else di[(di.onda == onda) & (di.clase == cl)]
            if g.empty:
                continue
            print(f"   {onda:>4} {cl:>5} {len(g):>7} {int((g.inicio_rel_s <= 0).sum()):>13} "
                  f"{int((g.inicio_rel_s > 0).sum()):>15} {g.aviso_rel_s.median():>31.1f} {g.iou.median():>8.2f}")
        print()

    pd.DataFrame(filas).to_csv(os.path.join(lib.DIR_SALIDA, "recorte_ruido.csv"), index=False)
    di.to_csv(os.path.join(lib.DIR_SALIDA, "recorte_inicio_onda.csv"), index=False)
    print(f"Tablas en {lib.DIR_SALIDA}/recorte_ruido.csv y recorte_inicio_onda.csv")


if __name__ == "__main__":
    main()
