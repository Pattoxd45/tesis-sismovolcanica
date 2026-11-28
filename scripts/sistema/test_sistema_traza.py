#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para sistema_traza.py: ventana centrada, imagenes, etiquetas, metricas y la evaluacion
completa con dos modelos de pesos al azar guardados como los de folds_cnn.py (sin PhaseNet ni datos reales).

Uso: python test_sistema_traza.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)


def caso(nombre, ok):
    print(f"  {'OK   ' if ok else 'FALLA'} {nombre}")
    return bool(ok)


def main():
    try:
        import torch
        import sistema_traza as st
    except ImportError as e:
        print(f"  (sin {e.name}: se omiten los casos)")
        return
    from scipy.signal import butter

    todo = True
    n = st.N_VENTANA
    todo &= caso("ventana centrada: al medio, contra el inicio y el fin del tramo, y recorte mas largo que la ventana",
                 st.ventana_centrada(50000, 52000, 0, 100000) == 51000 - n // 2
                 and st.ventana_centrada(1000, 2000, 0, 100000) == 0
                 and st.ventana_centrada(99000, 99500, 0, 100000) == 100000 - n
                 and st.ventana_centrada(40000, 80000, 0, 100000) == 40000)

    rng = np.random.default_rng(0)
    lim = [(0, 120000), (120000, 240000)]
    t = np.arange(240000) / st.FS
    crudo = {c: rng.normal(0, 1, t.size) + 50 * np.sin(2 * np.pi * 6 * t) * (((t > 300) & (t < 320)) | ((t > 1500) & (t < 1530)))
             for c in "ZNE"}
    # evento VT y LP del catalogo; recortes: uno sobre cada evento, uno que lo toca poco y uno sin evento
    eventos = pd.DataFrame(dict(ini=[30000, 150000], fin=[32000, 153000], clase=["VT", "LP"], incluido=True,
                                estado_crudo="coincide"))
    ivs = [(29900, 32100), (150200, 152800), (152500, 160000), (200000, 201000)]
    sos = butter(st.er.ORDEN_FILTRO, list(st.er.BANDA_FILTRO), btype="bandpass", fs=st.FS, output="sos")
    rec, fij, info = st.imagenes(crudo, ivs, lim, sos, 60.0)
    todo &= caso("imagenes: recorte y tramo fijo de 231 filas, ventana dentro de su tramo",
                 all(im.shape[0] == 231 and im.shape[2] == 3 for im in rec + fij)
                 and all(lim[1][0] <= i["ventana_ini"] <= lim[1][1] - n for i, (a, _) in zip(info, ivs) if a >= lim[1][0]))

    etq = st.etiquetas(eventos, ivs)
    todo &= caso("etiquetas como en el entrenamiento: VT, LP, fuera y sin caja",
                 etq.etiqueta.tolist() == ["VT", "LP", "fuera", "sin caja"])

    ec = st.ec
    pred = np.array([ec.CLASES.index("VT"), ec.CLASES.index("LP"), 0, ec.CLASES.index("TR")])
    m = st.metricas(pred, etq, eventos, ivs, horas=2.0)
    m2 = st.metricas(np.array([ec.CLASES.index("LP")] * 3 + [ec.CLASES.index("rechazo")]), etq, eventos, ivs, horas=2.0)
    todo &= caso("metricas: bien clasificados dan sensibilidad completa 1; la falsa alarma volcanica cuenta por hora",
                 m["VT_completo"] == 1 and m["LP_completo"] == 1 and m["volcanico_completo"] == 1
                 and m["falsas_por_hora"] == 0.5 and m["falsas_volcanicas_por_hora"] == 0.5
                 and m2["VT_completo"] == 0 and m2["LP_completo"] == 1 and m2["falsas_volcanicas_por_hora"] == 0
                 and np.isnan(m["TR_completo"]))

    d = tempfile.mkdtemp()
    for fold in (1, 2):
        red = ec.construir("efficientnet_b0", 4, False, 0.3)[0]
        nombre = os.path.join(d, f"efficientnet_b0_fold{fold}_x")
        torch.save(red.state_dict(), nombre + ".pt")
        cfg = dict(modelo="efficientnet_b0", conjunto="recorte", fold=fold, dropout=0.3, tamano="estirar",
                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], probabilidad="softmax")
        json.dump(dict(configuracion=cfg, test={}), open(nombre + ".json", "w"))
    modelos = st.modelos_guardados(d)
    etq2, filas, _ = st.evaluar(crudo, lim, ivs, eventos, eventos, 2.0, modelos, sos, 60.0, torch.device("cpu"))
    r = st.resumir(filas)
    todo &= caso("evaluacion completa con dos modelos: una prediccion por modelo y recorte, y resumen por red",
                 len(filas) == 2 and {"efficientnet_b0_recorte_fold1", "efficientnet_b0_recorte_fold2"} <= set(etq2.columns)
                 and len(r) == 4 and set(r.estadistico) == {"promedio", "minimo", "maximo", "desviacion"})

    print("TODOS PASAN" if todo else "HAY FALLAS")


if __name__ == "__main__":
    main()
