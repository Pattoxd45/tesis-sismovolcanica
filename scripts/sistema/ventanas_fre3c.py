#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deteccion y recorte de FRE procesando la traza en ventanas de 5 minutos: PhaseNet, con la
configuracion elegida, corre sobre cada ventana; los avisos repetidos entre ventanas se juntan con
la misma regla de 5 s que P y S, y el resultado se compara con el de la traza completa.

Uso: python ventanas_fre3c.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "fre3c"))
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
sys.path.insert(0, os.path.join(AQUI, "..", "ventanas"))
import fre3c_lib as lib
import anotar_fre3c as an
import barrido as br
import interseccion_fre3c as ix
import recorte_fre3c as rc
import recorte_detalle_fre3c as rd

FS = lib.FS
# 1-15 Hz: la banda que se mantuvo (1-45 Hz no mejora a ningun detector)
FILTRO = "f1_15"
FLUJO = "PhaseNet P y S"
DIR_SALIDA = os.path.join(lib.DIR_TESIS, "resultados", "sistema")


def avisos_de_ventana(ann, up, us):
    """Intervalo y pico de cada aviso de P y de S de una ventana, en muestras de la traza completa."""
    filas = []
    for tr in ann:
        fase = tr.stats.channel.split("_")[-1]
        if fase not in ("P", "S"):
            continue
        if abs(tr.stats.sampling_rate - FS) > 1e-6:
            raise SystemExit(f"DETENIDO: la anotacion {tr.id} viene a {tr.stats.sampling_rate} Hz")
        off = int(round(tr.stats.starttime.timestamp * FS))
        ini, fin, pico, valor = lib.disparos(np.asarray(tr.data), up if fase == "P" else us)
        for a, b, p, v in zip(ini, fin, pico, valor):
            filas.append(dict(fase=fase, ini=off + int(a), fin=off + int(b), pico=off + int(p), valor=float(v)))
    return filas


def juntar(av):
    """Intervalos para la interseccion y avisos para el recorte, con las reglas de 5 s ya en uso."""
    if av.empty:
        vacio = np.array([], dtype=np.int64)
        return vacio, vacio, vacio
    pi, pf = ix.agrupar_intervalos(av.ini.to_numpy(), av.fin.to_numpy())
    avisos = lib.agrupar_anclado(av.pico.to_numpy(), rc.TOL)
    return pi, pf, avisos


def resumen(nombre, res, inc, horas):
    fc, ft = lib.resumir(res, inc, horas["todos"])
    c = {f["clase"]: f for f in fc}
    print(f"   {nombre:<18} VT {c['VT']['k']}/{c['VT']['n']}  LP {c['LP']['k']}/{c['LP']['n']}  "
          f"TR {c['TR']['k']}/{c['TR']['n']}  volcanicos {c['volcanico']['k']}/{c['volcanico']['n']} = "
          f"{c['volcanico']['sens']:.3f}  {ft['fp_h']:.1f} FP/h")


def main():
    with open(os.path.join(lib.DIR_SALIDA, "configuracion_elegida.json")) as f:
        cfg = json.load(f)[FLUJO]
    if not cfg.get("por_defecto"):
        raise SystemExit("DETENIDO: la referencia usa las curvas guardadas, que son de las ventanas por defecto")
    up, us = cfg["P_threshold"], cfg["S_threshold"]
    kw = dict(overlap=int(cfg["overlap"]), blinding=(int(cfg["blinding"]), int(cfg["blinding"])),
              stacking=cfg["stacking"])

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zona = ix.unir(con_senal.ini.to_numpy(), con_senal.fin.to_numpy())
    z = np.load(lib.ruta_datos(FILTRO))
    valida = z["valida"]
    datos = {c: z[c] for c in ("Z", "N", "E")}
    horas = ix.horas_por_tramo(valida)
    ventanas = br.generar_ventanas(lib.N_TOTAL, lib.CORTES)

    print(f"1. PHASENET POR VENTANA (P_threshold {up}, S_threshold {us}; overlap {kw['overlap']}, "
          f"blinding {kw['blinding'][0]}, {kw['stacking']})")
    print(f"   {len(ventanas)} ventanas de {br.VENTANA_S:.0f} s con paso de {br.PASO_S:.0f} s, sin cruzar de un tramo a otro")
    m = an.cargar_modelo("phasenet")
    filas, tiempos = [], []
    for k, (a, b) in enumerate(ventanas):
        t = time.perf_counter()
        ann = m.annotate(an.stream_tramo(datos, a, b), **kw)
        tiempos.append(time.perf_counter() - t)
        filas += [dict(ventana=k, **f) for f in avisos_de_ventana(ann, up, us)]
    av = pd.DataFrame(filas)
    print(f"   {len(av)} avisos en total; PhaseNet demora {np.median(tiempos):.3f} s por ventana en mediana "
          f"({np.sum(tiempos):.1f} s en total, en CPU)")

    pi, pf, o_v = juntar(av)
    o_v = o_v[valida[o_v]]
    curvas = dict(np.load(lib.ruta_anot("phasenet", FILTRO)))
    qi, qf = ix.intervalos_ps(curvas, up, us, {})
    o_c = rd.onsets_ps(curvas, up, us, valida)

    print("\n2. DETECCION POR INTERSECCION")
    resumen("traza completa", ix.evaluar_interseccion(qi, qf, inc, zona, valida), inc, horas)
    resumen("ventanas de 5 min", ix.evaluar_interseccion(pi, pf, inc, zona, valida), inc, horas)

    print("\n3. AVISOS: VENTANAS CONTRA TRAZA COMPLETA (a 5 s o menos, uno a uno)")
    asig, usado = lib.emparejar(o_c, o_v, rc.TOL)
    print(f"   traza completa {o_c.size}, ventanas {o_v.size}; en ambas {int((asig >= 0).sum())}; "
          f"solo en la traza completa {int((asig < 0).sum())}; solo en las ventanas {int((~usado).sum())}")

    print("\n4. RECORTE POR ENERGIA CON CORTE (IoU con el catalogo)")
    env = rc.envolvente(datos["Z"], datos["N"], datos["E"])
    ivs = {}
    for nombre, o in (("traza completa", o_c), ("ventanas de 5 min", o_v)):
        ivs[nombre] = rc.recortes(o, env, valida, rc.K_RUIDO, corte=True)
        r = rd.medir(ivs[nombre], inc)
        print(f"   {nombre:<18} {r['recortes']} recortes; volcanicos que toca {r['toca']}, IoU >= 0,3: {r['k30']}, "
              f"IoU >= 0,5: {r['k50']}, IoU mediano {r['iou_mediano']:.2f}; TR con IoU >= 0,3: {r['tr_k30']}")

    print("\n5. RECORTES QUE CABEN ENTEROS EN UNA VENTANA DE 5 MIN")
    rv = ivs["ventanas de 5 min"]
    marg = int(rc.MARGEN_PREVIO_S * FS)
    ruido = int(rc.RUIDO_S[0] * FS)
    cabe = sum(1 for a, b in rv if br.ventanas_que_contienen(a, b, ventanas))
    # el periodo de ruido empieza 120 s antes del aviso, que esta 2,5 s despues del inicio del recorte
    cabe_ruido = sum(1 for a, b in rv if br.ventanas_que_contienen(a + marg - ruido, b, ventanas))
    print(f"   de {len(rv)} recortes, {cabe} caben enteros en alguna ventana; contando el periodo de ruido, {cabe_ruido}")

    os.makedirs(DIR_SALIDA, exist_ok=True)
    av.to_csv(os.path.join(DIR_SALIDA, "ventanas_avisos.csv"), index=False)
    pd.DataFrame(rv, columns=["ini", "fin"]).assign(dur_s=lambda d: (d.fin - d.ini) / FS).to_csv(
        os.path.join(DIR_SALIDA, "ventanas_recortes.csv"), index=False)
    print(f"\nTablas en {DIR_SALIDA}")


if __name__ == "__main__":
    main()
