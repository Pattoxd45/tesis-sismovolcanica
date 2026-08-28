#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barrido_eqt_nivel2.py
=====================

Cierra el ultimo hueco de la etapa de deteccion: barre el umbral de
EQTransformer sobre el NIVEL 2 (calidad del recorte) para compararlo de
forma pareja contra STA/LTA, que ya tenia sus parametros optimizados.

POR QUE IMPORTA: el detector existe para recortar eventos que despues
clasifica el VGG16, asi que la calidad del recorte es el criterio
decisivo. Hasta ahora STA/LTA ganaba ese eje contra un EQTransformer
corriendo con un umbral puesto a ojo, lo que no es una comparacion.

ESTRATEGIA DE COSTO: la inferencia se hace UNA vez por modo y se guarda
la curva Detection completa en disco (~73 MB). El barrido de umbrales
despues es aritmetica sobre esa curva y es instantaneo.

PhaseNet no entra: no produce canal Detection, no delimita eventos, y esa
ausencia es en si misma un resultado del trabajo.

Uso:
    python barrido_eqt_nivel2.py
    python barrido_eqt_nivel2.py --modos ceros
    python barrido_eqt_nivel2.py --recalcular
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_detectores import (FS, CANALES, CORTES, GUARDA_S, NPY, CSV,
                                SALIDA, CLASES, Intervalo, segmentos,
                                _stream, binaria_a_intervalos, fusionar_canal,
                                coincidencia, metricas)

CACHE = os.path.join(SALIDA, "cache_detection")

UMBRALES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
FUSION_S = [5.0, 15.0, 30.0]   # 30 s: los tremores duran 88 s de mediana
MIN_EST = [1, 2]
DUR_MIN_S = 3.0
IOU_MIN = 0.30


def curvas_detection(modo, canales, a, recalcular=False):
    """Corre EQTransformer una vez y cachea la curva Detection por canal."""
    os.makedirs(CACHE, exist_ok=True)
    ruta = os.path.join(CACHE, f"detection_eqt_{modo}.npz")
    if os.path.exists(ruta) and not recalcular:
        print(f"  usando cache: {ruta}")
        z = np.load(ruta)
        return {int(k): z[k] for k in z.files}

    import seisbench.models as sbm
    modelo = sbm.EQTransformer.from_pretrained("original")
    modelo.eval()

    n = a.shape[-1]
    curvas = {}
    for ch in canales:
        print(f"    EQT/{modo} canal {ch} ...", flush=True)
        curva = np.zeros(n, dtype=np.float32)
        for ini, fin in segmentos(n):
            x = np.asarray(a[ch, ini:fin], dtype=np.float64)
            if np.allclose(x, 0):
                continue
            ann = modelo.annotate(_stream(x, ini, modo, estacion=f"S{ch}"))
            tr = next((t for t in ann
                       if t.stats.channel.endswith("Detection")), None)
            if tr is None:
                continue
            off = int(round(tr.stats.starttime.timestamp * FS))
            lo, hi = max(ini, off), min(fin, off + tr.stats.npts)
            if hi > lo:
                curva[lo:hi] = tr.data[lo - off: hi - off]
        curvas[ch] = curva

    np.savez_compressed(ruta, **{str(k): v for k, v in curvas.items()})
    print(f"  curvas guardadas en {ruta}")
    return curvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modos", nargs="+", default=["ceros", "duplicar"])
    ap.add_argument("--canales", nargs="+", type=int, default=CANALES)
    ap.add_argument("--recalcular", action="store_true")
    args = ap.parse_args()

    a = np.load(NPY, mmap_mode="r")
    df_cat = pd.read_csv(CSV)
    n = a.shape[-1]

    valida = np.ones(n, dtype=bool)
    g = int(GUARDA_S * FS)
    for c in CORTES:
        valida[max(0, c - g):min(n, c + g)] = False
    catalogo = [Intervalo(int(r.idx_start), int(r.idx_end), r.event_type)
                for r in df_cat.itertuples()]
    catalogo = [e for e in catalogo if valida[e.ini] and valida[e.fin - 1]]
    horas = float(valida.sum()) / FS / 3600.0

    print(f"Catalogo: {len(catalogo)} eventos | {horas:.2f} h validas")
    print(f"Canales : {args.canales}\n")

    filas = []
    for modo in args.modos:
        print(f">>> EQTransformer | componentes={modo}")
        curvas = curvas_detection(modo, args.canales, a, args.recalcular)
        picos = {ch: float(c.max()) for ch, c in curvas.items()}
        print(f"    maximo de Detection por canal: "
              + ", ".join(f"ch{ch}={v:.2f}" for ch, v in picos.items()))

        for umbral, fus_s, k in itertools.product(UMBRALES, FUSION_S,
                                                  MIN_EST):
            por_canal = {}
            for ch, curva in curvas.items():
                ivs = [(iv.ini, iv.fin)
                       for iv in binaria_a_intervalos(curva >= umbral)]
                por_canal[ch] = {"intervalos":
                                 fusionar_canal(ivs, int(fus_s * FS))}
            det = coincidencia(por_canal, valida, k, tol_s=5.0,
                               dur_min_s=DUR_MIN_S)
            if not det:
                continue
            r, *_ = metricas(catalogo, det, horas, IOU_MIN)
            filas.append(dict(
                modelo="eqtransformer", modo=modo, umbral=umbral,
                fusion_s=fus_s, min_est=k, n_det=r["n_detecciones"],
                recall=r["recall"], precision=r["precision"], f1=r["f1"],
                fp_h=r["fp_por_hora"], iou=r["iou_medio"],
                onset_mae=r["onset_mae_s"],
                offset_mae=r["offset_mae_s"],
                dur_sesgo=r["duracion_sesgo_s"],
                **{f"rec_{c}": r["por_clase"].get(c, {}).get(
                    "recall", float("nan")) for c in CLASES}))

    if not filas:
        print("\nNo se genero ninguna fila.")
        return

    tabla = pd.DataFrame(filas)
    destino = os.path.join(SALIDA, "barrido_eqt_nivel2.csv")
    tabla.to_csv(destino, index=False)

    cols = ["modo", "umbral", "fusion_s", "min_est", "n_det", "recall",
            "precision", "f1", "fp_h", "iou", "dur_sesgo"]
    fmt = lambda v: f"{v:.3f}"

    print("\n" + "=" * 92)
    print("  EQTRANSFORMER NIVEL 2 - MEJORES 12 POR F1")
    print("=" * 92)
    print(tabla.sort_values("f1", ascending=False)[cols].head(12)
          .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 92)
    print("  MEJOR IoU (calidad del recorte, con recall > 0,30)")
    print("=" * 92)
    buenos = tabla[tabla.recall > 0.30].sort_values("iou", ascending=False)
    print(buenos[cols].head(8).to_string(index=False, float_format=fmt)
          if len(buenos) else "  ninguna combinacion supera recall 0,30")

    # --- comparacion contra STA/LTA -------------------------------
    ruta_st = os.path.join(SALIDA, "barrido_stalta.csv")
    if os.path.exists(ruta_st):
        st = pd.read_csv(ruta_st)
        mejor_st = st.sort_values("f1", ascending=False).iloc[0]
        mejor_eq = tabla.sort_values("f1", ascending=False).iloc[0]

        print("\n" + "=" * 92)
        print("  NIVEL 2 - COMPARACION PAREJA (cada uno en su mejor F1)")
        print("=" * 92)
        print(f"  {'metrica':<22}{'STA/LTA':>14}{'EQTransformer':>16}")
        print("  " + "-" * 52)
        for etiq, kst, keq in [
                ("recall", "recall", "recall"),
                ("precision", "precision", "precision"),
                ("F1", "f1", "f1"),
                ("falsos por hora", "fp_h", "fp_h"),
                ("IoU medio", "iou", "iou"),
                ("error onset MAE [s]", "onset_mae", "onset_mae"),
                ("sesgo duracion [s]", "dur_sesgo", "dur_sesgo")]:
            print(f"  {etiq:<22}{mejor_st[kst]:>14.3f}{mejor_eq[keq]:>16.3f}")

        print(f"\n  {'clase':<10}{'STA/LTA':>12}{'EQTransformer':>16}")
        print("  " + "-" * 38)
        for c in CLASES:
            print(f"  {c:<10}{mejor_st.get('rec_' + c, float('nan')):>12.3f}"
                  f"{mejor_eq.get('rec_' + c, float('nan')):>16.3f}")
    else:
        print(f"\n  (no se encontro {ruta_st}; corre barrido_stalta.py "
              "para la comparacion)")

    print(f"\nTabla completa en {destino}")
    print("""
QUE DECIDIR CON ESTO
  Si STA/LTA sigue ganando el Nivel 2 despues de que EQTransformer fue
  barrido igual de a fondo, la conclusion es solida: los modelos
  entrenados sobre sismos tectonicos localizan mejor el instante pero
  delimitan peor los eventos volcanicos sostenidos.

  Si EQTransformer alcanza a STA/LTA con un umbral mas bajo, entonces el
  resultado anterior era un artefacto del umbral y hay que corregirlo
  antes de escribir el capitulo.

  Cualquiera de los dos casos es un resultado publicable. Lo que no seria
  defendible es comparar un detector calibrado contra uno sin calibrar.
""")


if __name__ == "__main__":
    main()
