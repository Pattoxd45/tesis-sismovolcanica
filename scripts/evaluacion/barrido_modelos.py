#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barrido_modelos.py
==================

Barre los umbrales de PhaseNet y EQTransformer para que la comparacion
contra STA/LTA sea justa.

MOTIVO: STA/LTA tiene sus parametros optimizados sobre 324 combinaciones,
mientras que los modelos profundos corrian con un umbral de confianza
elegido a ojo. Comparar un detector calibrado contra dos sin calibrar no
es una comparacion, es un artefacto del ajuste.

ESTRATEGIA DE COSTO: la inferencia (lo caro) se hace UNA vez por
combinacion modelo x modo, con umbrales muy bajos para capturar todos los
picks posibles junto con su confianza. Los picks se guardan en disco. El
barrido de umbrales despues es aritmetica sobre esa tabla y es instantaneo.
Reejecutar el script reutiliza la cache.

Salida: curva precision-recall del Nivel 1 (deteccion del momento) para
cada modelo y modo de relleno de componentes.

Uso:
    python barrido_modelos.py                      # todo
    python barrido_modelos.py --modelos phasenet
    python barrido_modelos.py --recalcular         # ignora la cache
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_detectores import (FS, CANALES, CORTES, GUARDA_S, NPY, CSV,
                                SALIDA, CLASES, Intervalo, segmentos,
                                _stream, coincidencia_onsets, metricas_onset)

CACHE = os.path.join(SALIDA, "cache_picks")

# Umbrales muy bajos en la inferencia: se filtran despues por confianza.
KWARGS_BAJOS = {
    "phasenet": dict(P_threshold=0.02, S_threshold=0.02),
    "eqtransformer": dict(detection_threshold=0.02, P_threshold=0.02,
                          S_threshold=0.02),
}

UMBRALES_CONF = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75]
MIN_EST = [1, 2, 3]
TOL_REPORTE_S = 5.0


def extraer_picks(nombre, modo, canales, a, recalcular=False):
    """Corre el modelo una vez y devuelve un DataFrame de picks."""
    os.makedirs(CACHE, exist_ok=True)
    ruta = os.path.join(CACHE, f"picks_{nombre}_{modo}.csv")
    if os.path.exists(ruta) and not recalcular:
        print(f"  usando cache: {ruta}")
        return pd.read_csv(ruta)

    import seisbench.models as sbm
    clase = {"phasenet": sbm.PhaseNet, "eqtransformer": sbm.EQTransformer}
    modelo = clase[nombre].from_pretrained("original")
    modelo.eval()

    n = a.shape[-1]
    filas = []
    for ch in canales:
        print(f"    {nombre}/{modo} canal {ch} ...", flush=True)
        for ini, fin in segmentos(n):
            x = np.asarray(a[ch, ini:fin], dtype=np.float64)
            if np.allclose(x, 0):
                continue
            st = _stream(x, ini, modo, estacion=f"S{ch}")
            try:
                out = modelo.classify(st, **KWARGS_BAJOS[nombre])
            except TypeError:
                out = modelo.classify(st)
            for p in getattr(out, "picks", []):
                t = getattr(p, "peak_time", None)
                if t is None:
                    continue
                filas.append(dict(
                    canal=ch,
                    muestra=int(round(t.timestamp * FS)),
                    fase=str(getattr(p, "phase", "?")),
                    conf=float(getattr(p, "peak_value", 0.0) or 0.0)))

    df = pd.DataFrame(filas)
    df.to_csv(ruta, index=False)
    print(f"  {len(df)} picks guardados en {ruta}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+",
                    default=["phasenet", "eqtransformer"])
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
    for nombre in args.modelos:
        for modo in args.modos:
            print(f">>> {nombre} | componentes={modo}")
            picks = extraer_picks(nombre, modo, args.canales, a,
                                  args.recalcular)
            if picks.empty:
                print("    sin picks, se omite")
                continue
            print(f"    confianza: min={picks.conf.min():.3f} "
                  f"mediana={picks.conf.median():.3f} "
                  f"max={picks.conf.max():.3f}")

            for umbral in UMBRALES_CONF:
                sel = picks[picks.conf >= umbral]
                if sel.empty:
                    continue
                por_canal = {ch: {"onsets": sorted(gr.muestra.tolist())}
                             for ch, gr in sel.groupby("canal")}
                for k in MIN_EST:
                    onsets = coincidencia_onsets(por_canal, valida, k)
                    if not onsets:
                        continue
                    r = metricas_onset(catalogo, onsets, horas,
                                       [TOL_REPORTE_S])
                    d = r[f"tol_{TOL_REPORTE_S:g}s"]
                    rec, pre = d["recall"], d["precision"]
                    filas.append(dict(
                        modelo=nombre, modo=modo, umbral=umbral, min_est=k,
                        n_picks=len(sel), n_onsets=len(onsets),
                        recall=rec, precision=pre,
                        f1=2 * rec * pre / (rec + pre) if rec + pre else 0.0,
                        falsos_h=d["onsets_falsos_por_hora"],
                        err_mae=d["error_mae_s"],
                        **{f"rec_{c}": d["por_clase"].get(c, {}).get(
                            "recall", float("nan")) for c in CLASES}))

    if not filas:
        print("\nNo se genero ninguna fila.")
        return

    tabla = pd.DataFrame(filas)
    os.makedirs(SALIDA, exist_ok=True)
    destino = os.path.join(SALIDA, "barrido_modelos.csv")
    tabla.to_csv(destino, index=False)

    cols = ["modelo", "modo", "umbral", "min_est", "n_onsets", "recall",
            "precision", "f1", "falsos_h", "err_mae"]
    fmt = lambda v: f"{v:.3f}"

    print("\n" + "=" * 88)
    print(f"  MEJORES 12 POR F1  (tolerancia +-{TOL_REPORTE_S:g} s)")
    print("=" * 88)
    print(tabla.sort_values("f1", ascending=False)[cols].head(12)
          .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 88)
    print("  MEJOR RECALL CON MENOS DE 10 FALSOS/h")
    print("=" * 88)
    lim = tabla[tabla.falsos_h < 10].sort_values("recall", ascending=False)
    print(lim[cols].head(10).to_string(index=False, float_format=fmt)
          if len(lim) else "  ninguna combinacion cumple")

    print("\n" + "=" * 88)
    print("  CURVA PRECISION-RECALL POR MODELO Y MODO (min_est=2)")
    print("=" * 88)
    for (mo, md), gr in tabla[tabla.min_est == 2].groupby(["modelo", "modo"]):
        print(f"\n  {mo} | {md}")
        print(f"    {'umbral':>8}{'recall':>9}{'precision':>11}"
              f"{'falsos/h':>10}")
        for _, r in gr.sort_values("umbral").iterrows():
            print(f"    {r.umbral:>8.2f}{r.recall:>9.3f}"
                  f"{r.precision:>11.3f}{r.falsos_h:>10.2f}")

    print(f"\nTabla completa en {destino}")
    print("""
COMO USAR ESTO
  El punto de operacion debe elegirse con el MISMO criterio para los tres
  detectores. Si para STA/LTA se uso "mejor F1", hay que usar "mejor F1"
  aqui tambien; si se prefiere "mejor recall bajo un techo de falsas
  alarmas", ese criterio se aplica a los tres. Reportar cada detector en
  su mejor punto segun criterios distintos invalida la comparacion.

  La curva completa es mejor evidencia que cualquier punto unico: muestra
  si un detector domina al otro en todo el rango o solo en parte de el.
""")


if __name__ == "__main__":
    main()
