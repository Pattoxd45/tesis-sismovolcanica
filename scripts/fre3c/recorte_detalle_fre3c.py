#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recorte por energia de PhaseNet en detalle: efecto del periodo de ruido y del percentil, si
el recorte alcanza el inicio del evento segun la onda (P o S) que dio el aviso, y cuantos
recortes que venian solo de la S se corrigen al bajar P_threshold.

Uso: python recorte_detalle_fre3c.py [--umbral 0.10] [--p P_threshold] [--s S_threshold] [--filtro f1_15]
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
P_MENORES = [0.02, 0.03, 0.05]  # solo se baja P_threshold; S_threshold queda como en la base
OTRAS = [(0.30, 0.10), (0.40, 0.05)]  # 0,95 con +-10 s y con interseccion


def medir(ivs, inc):
    pe = pd.DataFrame(iu.por_evento("x", ivs, inc))
    v = pe[pe.clase.isin(lib.VOLCANICAS)]
    tr = v[v.clase == "TR"]
    return dict(toca=int((v.iou > 0).sum()), k30=int((v.iou >= 0.3).sum()), k50=int((v.iou >= 0.5).sum()),
                iou_mediano=v[v.iou > 0].iou.median(), tr_k30=int((tr.iou >= 0.3).sum()),
                tr_iou_mediano=tr[tr.iou > 0].iou.median(), recortes=len(ivs))


def onsets_ps(curvas, up, us, valida):
    """Como rc.onsets_pn_ps, con P_threshold y S_threshold por separado."""
    partes = [lib.disparos_por_tramo(curvas["P"], up)[2], lib.disparos_por_tramo(curvas["S"], us)[2]]
    o = lib.agrupar_anclado(np.concatenate(partes), rc.TOL)
    return o[valida[o]]


def inicio_por_onda(curvas, up, us, env, valida, inc):
    """Por volcanico con recorte que lo toca: onda que dio el aviso y donde empieza el recorte."""
    ivs = rc.recortes(onsets_ps(curvas, up, us, valida), env, valida, rc.K_RUIDO, corte=True)
    pico_p = set(lib.disparos_por_tramo(curvas["P"], up)[2].tolist())
    pico_s = set(lib.disparos_por_tramo(curvas["S"], us)[2].tolist())
    m = int(rc.MARGEN_PREVIO_S * FS)
    cat = [ed.Intervalo(int(r.ini), int(r.fin), r.clase) for r in inc.itertuples()]
    pares, _, _ = ed.emparejar(cat, [ed.Intervalo(a, b) for a, b in ivs], iu.TOCA)
    filas = []
    for g, d, v in pares:
        aviso = d.ini + m
        onda = "P" if aviso in pico_p else ("S" if aviso in pico_s else "?")
        filas.append(dict(ev_ini=g.ini, clase=g.clase, onda=onda, iou=v, inicio_rel_s=(d.ini - g.ini) / FS,
                          aviso_rel_s=(aviso - g.ini) / FS))
    di = pd.DataFrame(filas)
    return ivs, di[di.clase.isin(lib.VOLCANICAS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.10)
    ap.add_argument("--p", type=float, default=None, help="P_threshold (por defecto, --umbral)")
    ap.add_argument("--s", type=float, default=None, help="S_threshold (por defecto, --umbral)")
    ap.add_argument("--filtro", default="f1_15", choices=list(lib.FILTROS))
    args = ap.parse_args()
    up = args.umbral if args.p is None else args.p
    us = args.umbral if args.s is None else args.s

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    z = np.load(lib.ruta_datos(args.filtro))
    valida = z["valida"]
    env = rc.envolvente(z["Z"], z["N"], z["E"])
    pn = np.load(lib.ruta_anot("phasenet", args.filtro))
    curvas = {k: pn[k] for k in ("P", "S")}
    o = onsets_ps(curvas, up, us, valida)
    n_volc = int(inc.clase.isin(lib.VOLCANICAS).sum())

    print(f"RECORTE POR ENERGIA CON CORTE EN DETALLE (PhaseNet, P_threshold {up:.2f}, S_threshold {us:.2f}, "
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
    _, di = inicio_por_onda(curvas, up, us, env, valida, inc)
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

    # los recortes que vienen solo de la S con la base se siguen con P_threshold mas bajo
    solo_s = di[di.onda == "S"]
    tarde = set(solo_s[solo_s.inicio_rel_s > 0].ev_ini)
    print(f"3. RECORTES QUE VIENEN SOLO DE LA S CON P_threshold {up:.2f} y S_threshold {us:.2f}: "
          f"{len(solo_s)}, de ellos {len(tarde)} empiezan despues del inicio del evento")
    print(f"   {'P_threshold':>11} {'S_threshold':>11} {'aviso por P':>11} {'ahora antes':>11} "
          f"{'sigue tarde':>11} {'sin recorte':>11} | {'todos: solo S':>13} {'solo S tarde':>12} "
          f"{'IoU>=0.3':>8} {'recortes':>8}")
    filas_p = []
    for p_, s_ in [(p, us) for p in P_MENORES if p < up] + [(up, us)] + OTRAS:
        ivs_c, dc = inicio_por_onda(curvas, p_, s_, env, valida, inc)
        sig = dc[dc.ev_ini.isin(tarde)]
        mm = medir(ivs_c, inc)
        fila = dict(P_threshold=p_, S_threshold=s_, base_tarde=len(tarde),
                    aviso_por_p=int((sig.onda == "P").sum()), ahora_antes=int((sig.inicio_rel_s <= 0).sum()),
                    sigue_tarde=int((sig.inicio_rel_s > 0).sum()), sin_recorte=len(tarde) - len(sig),
                    solo_s=int((dc.onda == "S").sum()),
                    solo_s_tarde=int(((dc.onda == "S") & (dc.inicio_rel_s > 0)).sum()),
                    k30=mm["k30"], recortes=mm["recortes"])
        filas_p.append(fila)
        print(f"   {p_:>11.2f} {s_:>11.2f} {fila['aviso_por_p']:>11} {fila['ahora_antes']:>11} "
              f"{fila['sigue_tarde']:>11} {fila['sin_recorte']:>11} | {fila['solo_s']:>13} "
              f"{fila['solo_s_tarde']:>12} {fila['k30']:>8} {fila['recortes']:>8}")
    print(f"   columnas a la izquierda: los {len(tarde)} eventos que empezaban tarde con la base; "
          "a la derecha: todos los volcanicos con esa combinacion")

    pd.DataFrame(filas).to_csv(os.path.join(lib.DIR_SALIDA, "recorte_ruido.csv"), index=False)
    di.to_csv(os.path.join(lib.DIR_SALIDA, "recorte_inicio_onda.csv"), index=False)
    pd.DataFrame(filas_p).to_csv(os.path.join(lib.DIR_SALIDA, "recorte_p_threshold.csv"), index=False)
    print(f"\nTablas en {lib.DIR_SALIDA}/recorte_ruido.csv, recorte_inicio_onda.csv y recorte_p_threshold.csv")


if __name__ == "__main__":
    main()
