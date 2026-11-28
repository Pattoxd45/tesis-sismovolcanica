#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recorte de referencia contra recorte automatico en la traza continua (OE4). Clasifica cada evento del
catalogo con su propio inicio y fin (recorte de referencia), con los mismos espectrogramas y los mismos
modelos que sistema_traza.py, y lo compara evento a evento con lo que logra el sistema (detectado, recortado
con IoU >= 0,3 y bien clasificado), leido de sistema_traza_recortes.csv. Por red y grupo de clases da el
acierto con cada recorte, la diferencia pareada (automatico menos referencia) y su intervalo del 95 % por
bootstrap sobre los eventos; lo mismo solo con los eventos cuyo recorte automatico calza (IoU >= 0,3).

Uso: python referencia_traza.py   (requiere haber corrido sistema_traza.py)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import sistema_traza as st

ec, er, lib, ed, iu = st.ec, st.er, st.lib, st.ed, st.iu
IOU_MIN = st.IOU_MIN
GRUPOS = lib.CLASES + ["volcanico"]
N_BOOTSTRAP = 10000


def grupo(clases, k):
    return np.isin(clases, lib.VOLCANICAS) if k == "volcanico" else clases == k


def pares_automaticos(inc, ivs):
    """Para cada evento, el recorte automatico que le toca (uno a uno, como en sistema_traza.py) y su IoU;
    -1 y 0 si ninguno lo toca."""
    cat = [ed.Intervalo(int(r.ini), int(r.fin), r.clase) for r in inc.itertuples()]
    det = [ed.Intervalo(int(a), int(b)) for a, b in ivs]
    pos = {id(d): k for k, d in enumerate(det)}
    pares, _, _ = ed.emparejar(cat, det, iu.TOCA)
    par = {id(g): (pos[id(d)], v) for g, d, v in pares}
    j = [par.get(id(g), (-1, 0.0)) for g in cat]
    return np.array([a for a, _ in j], dtype=int), np.array([v for _, v in j], dtype=float)


def evaluar_referencia(crudo, limites, inc, auto, modelos, sos, fija, disp):
    """Una fila por evento y modelo: prediccion con el recorte de referencia y la del sistema para el mismo
    evento (vacia si su recorte no calza); tambien el tiempo de las imagenes de referencia."""
    ivs_ref = [(int(a), int(b)) for a, b in zip(inc.ini, inc.fin)]
    t = time.perf_counter()
    rec, fij, info = st.imagenes(crudo, ivs_ref, limites, sos, fija)
    t_img = time.perf_counter() - t
    j, iou = pares_automaticos(inc, list(zip(auto.ini, auto.fin)))
    calza = (j >= 0) & (iou >= IOU_MIN)
    real = inc.clase.map(ec.MAPA).to_numpy()
    filas = []
    for (modelo, conjunto), lista in sorted(modelos.items()):
        for cfg, pesos in sorted(lista, key=lambda x: x[0]["fold"]):
            col = f"{modelo}_{conjunto}_fold{cfg['fold']}"
            p = st.clasificar(cfg, pesos, fij if conjunto.startswith("fijo") else rec, disp).argmax(1)
            ref = np.array([ec.CLASES[i] for i in p], dtype=object)
            pa = np.where(calza, auto[col].to_numpy()[np.maximum(j, 0)], "")
            filas.append(pd.DataFrame(dict(
                evento=np.arange(len(inc)), clase=inc.clase.to_numpy(), real=real, modelo=modelo,
                conjunto=conjunto, fold=cfg["fold"], calza=calza, iou=iou, pred_referencia=ref,
                pred_automatico=pa, referencia_bien=ref == real, automatico_bien=pa == real)))
    d = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
    return d, t_img, int(sum(x["cortado"] for x in info))


def metricas_modelo(d):
    """Para un modelo: F1 macro y sensibilidad por clase con el recorte de referencia, y acierto por grupo con
    cada recorte, en todos los eventos y en los que tienen un recorte automatico que calza."""
    num = {k: i for i, k in enumerate(ec.CLASES)}
    m = ec.metricas(ec.matriz(d.real.map(num).to_numpy(), d.pred_referencia.map(num).to_numpy()))
    out = dict(f1_macro_referencia=m["f1_macro_volcanicas"])
    for k in ec.CLASES:
        out[f"{k}_sensibilidad_referencia"] = m[k]["sensibilidad"]
    clases, calza = d.clase.to_numpy(), d.calza.to_numpy()
    for k in GRUPOS:
        for sufijo, mask in (("", grupo(clases, k)), ("_calza", grupo(clases, k) & calza)):
            out[f"{k}_referencia{sufijo}"] = float(d.referencia_bien[mask].mean()) if mask.any() else float("nan")
            out[f"{k}_automatico{sufijo}"] = float(d.automatico_bien[mask].mean()) if mask.any() else float("nan")
    return out


def bootstrap(x, n=N_BOOTSTRAP, semilla=0):
    """Intervalo del 95 % del promedio de x, sorteando eventos con reemplazo (percentiles 2,5 y 97,5)."""
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(semilla)
    medias = np.asarray(x, dtype=float)[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def comparar(d, n=N_BOOTSTRAP, semilla=0):
    """Por red, conjunto y grupo: acierto promedio de los 5 modelos con cada recorte, diferencia pareada
    (automatico menos referencia) con su intervalo, y eventos en que solo acierta uno de los dos (promedio
    entre modelos); en todos los eventos y en los que tienen recorte automatico que calza."""
    filas = []
    for (modelo, conjunto), g in d.groupby(["modelo", "conjunto"]):
        e = g.groupby("evento").agg(clase=("clase", "first"), calza=("calza", "first"),
                                    r=("referencia_bien", "mean"), a=("automatico_bien", "mean"))
        for k in GRUPOS:
            for eventos, mask in (("todos", grupo(e.clase.to_numpy(), k)),
                                  ("calza", grupo(e.clase.to_numpy(), k) & e.calza.to_numpy())):
                x = e[mask]
                dif = (x.a - x.r).to_numpy()
                gg = g[g.evento.isin(x.index)]
                solo_ref = (gg.referencia_bien & ~gg.automatico_bien).groupby(gg.fold).sum()
                solo_auto = (~gg.referencia_bien & gg.automatico_bien).groupby(gg.fold).sum()
                lo, hi = bootstrap(dif, n, semilla)
                filas.append(dict(modelo=modelo, conjunto=conjunto, grupo=k, eventos=eventos, n=len(x),
                                  referencia=x.r.mean() if len(x) else float("nan"),
                                  automatico=x.a.mean() if len(x) else float("nan"),
                                  diferencia=dif.mean() if len(x) else float("nan"), ic_inf=lo, ic_sup=hi,
                                  solo_referencia=float(solo_ref.mean()) if len(solo_ref) else 0.0,
                                  solo_automatico=float(solo_auto.mean()) if len(solo_auto) else 0.0))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=st.FOLDS, help="carpeta con los JSON y pesos de folds_cnn.py")
    ap.add_argument("--modelos", nargs="*", default=None, help="solo estas redes")
    ap.add_argument("--ventana-fija", type=float, default=60.0)
    ap.add_argument("--salida", default=st.SALIDA)
    args = ap.parse_args()
    from scipy.signal import butter

    ruta = os.path.join(args.salida, "sistema_traza_recortes.csv")
    if not os.path.exists(ruta):
        raise SystemExit(f"DETENIDO: falta {ruta}; hay que correr antes sistema_traza.py")
    auto = pd.read_csv(ruta, keep_default_na=False)
    modelos = st.modelos_guardados(args.folds, args.modelos)
    incompletos = {k: len(v) for k, v in modelos.items() if len(v) != 5}
    faltan = [f"{m}_{c}_fold{cfg['fold']}" for (m, c), v in modelos.items() for cfg, _ in v
              if f"{m}_{c}_fold{cfg['fold']}" not in auto.columns]
    if not modelos or incompletos or faltan:
        raise SystemExit(f"DETENIDO: modelos incompletos {incompletos} o sin resultado en sistema_traza: {faltan}")
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    print(f"{len(inc)} eventos del catalogo ({', '.join(f'{k} {v}' for k, v in inc.clase.value_counts().items())}); "
          f"{len(auto)} recortes automaticos; {disp}")

    crudo = st.crudo_alineado()
    sos = butter(er.ORDEN_FILTRO, list(er.BANDA_FILTRO), btype="bandpass", fs=st.FS, output="sos")
    d, t_img, cortados = evaluar_referencia(crudo, lib.LIMITES_TRAMO, inc, auto, modelos, sos, args.ventana_fija, disp)
    print(f"Imagenes de referencia: {t_img:.1f} s; {cortados} eventos mas largos que la ventana")

    filas = [dict(modelo=m, conjunto=c, fold=f, **metricas_modelo(g)) for (m, c, f), g in d.groupby(["modelo", "conjunto", "fold"])]
    por_modelo = pd.DataFrame(filas)
    prev = os.path.join(args.salida, "sistema_traza_modelos.csv")
    if os.path.exists(prev):
        s = pd.read_csv(prev).merge(por_modelo, on=["modelo", "conjunto", "fold"])
        dif = max(abs(s[f"{k}_completo"] - s[f"{k}_automatico"]).max() for k in GRUPOS)
        print(f"Control: acierto automatico igual a la sensibilidad completa de sistema_traza.py: "
              f"{'si' if dif < 1e-9 else 'NO'} (diferencia maxima {dif:.4f})")

    print("\nCon el recorte de referencia (promedio de los 5 modelos):")
    cols = ["f1_macro_referencia"] + [f"{k}_sensibilidad_referencia" for k in ec.CLASES]
    print(por_modelo.groupby(["modelo", "conjunto"])[cols].agg(["mean", "std"]).round(3).T.to_string())

    comp = comparar(d)
    for (modelo, conjunto), g in comp.groupby(["modelo", "conjunto"]):
        print(f"\n{modelo}, {conjunto}: acierto con cada recorte y diferencia automatico menos referencia [IC 95 %]")
        for eventos, h in g.groupby("eventos", sort=False):
            print("   " + ("todos los eventos (automatico = sensibilidad completa)" if eventos == "todos"
                           else "solo eventos cuyo recorte automatico calza (IoU >= 0,3)"))
            t = h.set_index("grupo")[["n", "referencia", "automatico", "diferencia", "ic_inf", "ic_sup",
                                      "solo_referencia", "solo_automatico"]]
            print(t.round(3).to_string())

    os.makedirs(args.salida, exist_ok=True)
    d.to_csv(os.path.join(args.salida, "referencia_traza_eventos.csv"), index=False)
    por_modelo.to_csv(os.path.join(args.salida, "referencia_traza_modelos.csv"), index=False)
    comp.to_csv(os.path.join(args.salida, "referencia_traza_comparacion.csv"), index=False)
    json.dump(dict(eventos=len(inc), imagenes_s=t_img, bootstrap=N_BOOTSTRAP, dispositivo=str(disp)),
              open(os.path.join(args.salida, "referencia_traza_tiempos.json"), "w"), indent=1)
    print(f"\nTablas en {args.salida} (referencia_traza_*.csv)")


if __name__ == "__main__":
    main()
