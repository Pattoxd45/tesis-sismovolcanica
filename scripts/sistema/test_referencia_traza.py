#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para referencia_traza.py: emparejamiento evento-recorte, intervalo por bootstrap,
comparacion pareada y la evaluacion completa con dos modelos de pesos al azar (sin PhaseNet ni datos reales).

Uso: python test_referencia_traza.py
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
        import referencia_traza as rt
    except ImportError as e:
        print(f"  (sin {e.name}: se omiten los casos)")
        return
    from scipy.signal import butter

    st, ec = rt.st, rt.ec
    todo = True
    inc = pd.DataFrame(dict(ini=[30000, 150000, 180000], fin=[32000, 153000, 181000], clase=["VT", "LP", "IC"]))
    ivs = [(29900, 32100), (152500, 160000), (200000, 201000)]
    j, iou = rt.pares_automaticos(inc, ivs)
    todo &= caso("emparejamiento: el VT con su recorte, el LP con uno que lo toca poco y el IC sin recorte",
                 j.tolist() == [0, 1, -1] and iou[0] > 0.9 and 0 < iou[1] < 0.3 and iou[2] == 0)

    lo, hi = rt.bootstrap(np.array([0.2] * 10))
    lo2, hi2 = rt.bootstrap(np.array([0.0] * 10 + [1.0] * 10))
    todo &= caso("bootstrap: sin variacion da un intervalo de ancho cero; con variacion contiene el promedio",
                 abs(lo - 0.2) < 1e-12 and abs(hi - 0.2) < 1e-12 and lo2 < 0.5 < hi2 and np.isnan(rt.bootstrap([])[0]))

    # dos modelos y tres eventos: referencia acierta todo; automatico acierta solo el VT
    filas = []
    for fold in (1, 2):
        filas.append(pd.DataFrame(dict(evento=[0, 1, 2], clase=["VT", "LP", "IC"], real=["VT", "LP", "rechazo"],
                                       modelo="m", conjunto="recorte", fold=fold, calza=[True, False, False],
                                       iou=[0.95, 0.1, 0.0], pred_referencia=["VT", "LP", "rechazo"],
                                       pred_automatico=["VT", "", ""], referencia_bien=True,
                                       automatico_bien=[True, False, False])))
    d = pd.concat(filas, ignore_index=True)
    c = rt.comparar(d, n=500).set_index(["grupo", "eventos"])
    m = rt.metricas_modelo(d[d.fold == 1])
    todo &= caso("comparacion pareada: diferencias, n y eventos en que solo acierta la referencia",
                 c.loc[("volcanico", "todos"), "n"] == 2 and c.loc[("volcanico", "todos"), "diferencia"] == -0.5
                 and c.loc[("volcanico", "calza"), "n"] == 1 and c.loc[("volcanico", "calza"), "diferencia"] == 0
                 and c.loc[("LP", "todos"), "solo_referencia"] == 1 and c.loc[("IC", "todos"), "automatico"] == 0
                 and m["volcanico_referencia"] == 1 and m["volcanico_automatico"] == 0.5
                 and m["VT_automatico_calza"] == 1 and np.isnan(m["LP_automatico_calza"]))

    rng = np.random.default_rng(0)
    lim = [(0, 120000), (120000, 240000)]
    t = np.arange(240000) / st.FS
    crudo = {k: rng.normal(0, 1, t.size) + 50 * np.sin(2 * np.pi * 6 * t) * ((t > 300) & (t < 320)) for k in "ZNE"}
    auto = pd.DataFrame(dict(ini=[a for a, _ in ivs], fin=[b for _, b in ivs]))
    dtmp = tempfile.mkdtemp()
    for fold in (1, 2):
        red = ec.construir("efficientnet_b0", 4, False, 0.3)[0]
        nombre = os.path.join(dtmp, f"efficientnet_b0_fold{fold}_x")
        torch.save(red.state_dict(), nombre + ".pt")
        cfg = dict(modelo="efficientnet_b0", conjunto="recorte", fold=fold, dropout=0.3, tamano="estirar",
                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], probabilidad="softmax")
        json.dump(dict(configuracion=cfg, test={}), open(nombre + ".json", "w"))
        auto[f"efficientnet_b0_recorte_fold{fold}"] = ["VT", "TR", "LP"]
    modelos = st.modelos_guardados(dtmp)
    sos = butter(st.er.ORDEN_FILTRO, list(st.er.BANDA_FILTRO), btype="bandpass", fs=st.FS, output="sos")
    d2, _, cortados = rt.evaluar_referencia(crudo, lim, inc, auto, modelos, sos, 60.0, torch.device("cpu"))
    todo &= caso("evaluacion completa: una fila por evento y modelo; el sistema solo cuenta el recorte que calza",
                 len(d2) == 6 and cortados == 0
                 and d2[d2.evento == 0].pred_automatico.tolist() == ["VT", "VT"]
                 and d2[d2.evento == 0].automatico_bien.all()
                 and (d2[d2.evento > 0].pred_automatico == "").all() and not d2[d2.evento > 0].automatico_bien.any())

    print("TODOS PASAN" if todo else "HAY FALLAS")


if __name__ == "__main__":
    main()
