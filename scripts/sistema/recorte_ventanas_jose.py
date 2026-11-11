#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recorte de las ventanas de 5 minutos de Jose con la configuracion elegida (PhaseNet P y S, recorte
por energia con corte) y comparacion con sus cajas: deteccion por interseccion e IoU por clase. Mide
dos formas de tomar el periodo de ruido del recorte: solo con la senal de la ventana, o tambien con
los 120 s anteriores. PhaseNet ve solo la ventana en las dos.

Uso: python recorte_ventanas_jose.py [--limite 50]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "fre3c"))
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
sys.path.insert(0, os.path.join(AQUI, "..", "winston"))
import fre3c_lib as lib
import anotar_fre3c as an
import evaluar_detectores as ed
import interseccion_fre3c as ix
import iou_fre3c as iu
import recorte_fre3c as rc
import ventanas_fre3c as vf
import extraer_ventanas_jose as ej
import extraer_previo_jose as ep

FS = lib.FS
FLUJO = "PhaseNet P y S"
# 1-15 Hz: la banda que se mantuvo (1-45 Hz no mejora a ningun detector)
FILTRO = "f1_15"
CAJAS = os.path.join(ej.DIR_JOSE, "metadata_dataset", "metadata", "bounding_boxes_para_entrenamiento.csv")
DIR_SALIDA = os.path.join(lib.DIR_TESIS, "resultados", "sistema")
VARIANTES = {"solo ventana": False, "con 120 s previos": True}


def procesar(x, desde, m, kw, up, us, sos):
    """x: Z, N y E sin filtrar; PhaseNet corre desde la muestra `desde` (inicio de la ventana) hasta el final.
    Devuelve los intervalos de P y S agrupados, los avisos y los recortes, en muestras de la ventana."""
    n = x["Z"].size
    f = {c: sosfiltfilt(sos, np.asarray(x[c], dtype=np.float64)).astype(np.float32) for c in ("Z", "N", "E")}
    av = pd.DataFrame(vf.avisos_de_ventana(m.annotate(an.stream_tramo(f, desde, n), **kw), up, us))
    pi, pf, o = vf.juntar(av)
    env = rc.envolvente(f["Z"], f["N"], f["E"])
    ivs = rc.recortes(o, env, np.ones(n, dtype=bool), rc.K_RUIDO, corte=True, limites=[(0, n)])
    # lo anterior a la ventana solo sirve para el filtro y el ruido: el recorte se lleva a la ventana
    ivs = [(max(a - desde, 0), b - desde) for a, b in ivs if b > desde]
    return pi - desde, pf - desde, o - desde, ivs


def evaluar(cajas, pi, pf, ivs, n):
    """Por caja: si un intervalo de P o S la cruza (uno a uno) y el IoU del recorte que le toca (uno a uno)."""
    inc = pd.DataFrame(dict(ini=cajas.ini.to_numpy(), fin=cajas.fin.to_numpy()))
    zona = ix.unir(inc.ini.to_numpy(), inc.fin.to_numpy())
    r = ix.evaluar_interseccion(pi, pf, inc, zona, np.ones(n, dtype=bool))
    cat = [ed.Intervalo(int(a), int(b), c) for a, b, c in zip(cajas.ini, cajas.fin, cajas.clase)]
    pares, _, sin_caja = ed.emparejar(cat, [ed.Intervalo(int(a), int(b)) for a, b in ivs], iu.TOCA)
    iou = {id(g): v for g, _, v in pares}
    return r["asignado"] >= 0, np.array([iou.get(id(g), 0.0) for g in cat]), len(sin_caja)


def cargar_cajas():
    c = pd.read_csv(CAJAS, encoding="utf-8-sig")
    c["ini"] = np.round(c.inicio_rel_s * FS).astype(np.int64)
    c["fin"] = np.round(c.fin_rel_s * FS).astype(np.int64)
    return {e: g.reset_index(drop=True) for e, g in c.groupby("ventana_evento_principal")}


def linea(t, nombre):
    out = []
    for clase in lib.CLASES + ["volcanico"]:
        x = t[t.clase.isin(lib.VOLCANICAS)] if clase == "volcanico" else t[t.clase == clase]
        if len(x):
            out.append(f"{clase} {int(x.detectada.sum())}/{len(x)} ({x.detectada.mean():.3f}), IoU>=0,3 "
                       f"{(x.iou >= 0.3).mean():.3f}, IoU>=0,5 {(x.iou >= 0.5).mean():.3f}")
    print(f"   {nombre}:\n      " + "\n      ".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N ventanas, para probar")
    args = ap.parse_args()

    with open(os.path.join(lib.DIR_SALIDA, "configuracion_elegida.json")) as f:
        cfg = json.load(f)[FLUJO]
    up, us = cfg["P_threshold"], cfg["S_threshold"]
    kw = dict(overlap=int(cfg["overlap"]), blinding=(int(cfg["blinding"]), int(cfg["blinding"])),
              stacking=cfg["stacking"])

    reg = pd.read_csv(ej.REGISTRO).set_index("evento")
    prev = pd.read_csv(ep.REGISTRO).set_index("evento")
    prev_ok = prev[(prev.con_dato_Z == 1) & (prev.con_dato_N == 1) & (prev.con_dato_E == 1)].index
    ev = [e for e in ep.completas(ej.REGISTRO) if e in prev_ok]
    if args.limite:
        ev = ev[:args.limite]
    cajas = cargar_cajas()
    print(f"{len(ev)} ventanas completas con sus {ep.PREVIO_S:.0f} s previos completos; PhaseNet P_threshold {up}, "
          f"S_threshold {us}, overlap {kw['overlap']}, blinding {kw['blinding'][0]}, {kw['stacking']}")

    m = an.cargar_modelo("phasenet")
    sos = lib.sos_filtro(lib.FILTROS[FILTRO])
    filas_caja, filas_rec, filas_ven, t_ini = [], [], [], time.time()
    for k, e in enumerate(ev, 1):
        w = np.load(os.path.join(ej.DIR_SALIDA, f"{e}_zne.npz"))
        p = np.load(os.path.join(ep.DIR_SALIDA, f"{e}_zne.npz"))
        cj = cajas[e]
        for var, con_previo in VARIANTES.items():
            x = {c: np.concatenate([p[c], w[c]]) if con_previo else w[c] for c in ("Z", "N", "E")}
            pi, pf, o, ivs = procesar(x, ep.N_PREVIO if con_previo else 0, m, kw, up, us, sos)
            det, iou, sin_caja = evaluar(cj, pi, pf, ivs, ej.N_VENTANA)
            filas_caja += [dict(variante=var, ventana=e, split=reg.loc[e, "split"], tipo=r.tipo_marca, clase=r.clase,
                                ini_s=r.inicio_rel_s, fin_s=r.fin_rel_s, detectada=bool(d), iou=float(v))
                           for r, d, v in zip(cj.itertuples(), det, iou)]
            filas_rec += [dict(variante=var, ventana=e, ini=a, fin=b) for a, b in ivs]
            filas_ven.append(dict(variante=var, ventana=e, recortes=len(ivs), sin_caja=sin_caja))
        if k % 500 == 0:
            print(f"  {k}/{len(ev)} ventanas en {(time.time() - t_ini) / 60:.1f} min", flush=True)

    t = pd.DataFrame(filas_caja)
    rec = pd.DataFrame(filas_rec, columns=["variante", "ventana", "ini", "fin"])
    ven = pd.DataFrame(filas_ven)
    print(f"\nTiempo: {(time.time() - t_ini) / 60:.1f} min para {len(ev)} ventanas y {len(VARIANTES)} variantes")
    for var in VARIANTES:
        x = t[t.variante == var]
        v = ven[ven.variante == var]
        print(f"\n{var.upper()}: {int(v.recortes.sum())} recortes ({v.recortes.mean():.2f} por ventana), "
              f"{int(v.sin_caja.sum())} sin caja de Jose; ventanas sin ningun recorte {int((v.recortes == 0).sum())}")
        linea(x[x.tipo == "PRINCIPAL"], "evento principal de cada ventana")
        linea(x, "todas las cajas (principales y extra)")
        temprano = x[(x.tipo == "PRINCIPAL") & (x.ini_s < ep.PREVIO_S)]
        linea(temprano, f"principal que empieza antes del segundo {ep.PREVIO_S:.0f} ({len(temprano)} ventanas)")

    os.makedirs(DIR_SALIDA, exist_ok=True)
    t.to_csv(os.path.join(DIR_SALIDA, "recorte_jose_cajas.csv"), index=False)
    rec.to_csv(os.path.join(DIR_SALIDA, "recorte_jose_recortes.csv"), index=False)
    ven.to_csv(os.path.join(DIR_SALIDA, "recorte_jose_ventanas.csv"), index=False)
    print(f"\nTablas en {DIR_SALIDA}")


if __name__ == "__main__":
    main()
