#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparar_detectores.py
======================

Cierra la comparacion de los tres detectores bajo un criterio unico.

PROBLEMA QUE RESUELVE: hasta ahora STA/LTA tenia su punto de operacion
elegido optimizando el Nivel 2 (calidad del recorte), mientras que
PhaseNet y EQTransformer se barrieron sobre el Nivel 1 (deteccion del
momento). Comparar detectores optimizados con criterios distintos no es
una comparacion. Este script barre STA/LTA con metricas de onset, une el
resultado con barrido_modelos.csv y selecciona el punto de operacion de
los tres con la MISMA regla.

Produce:
  - resultados/comparacion_nivel1.csv   tabla unificada
  - resultados/frontera_pareto.csv      frontera precision-recall
  - resultados/curva_pr_nivel1.png      figura para la tesis
  - tabla final impresa en consola

Uso:
    python comparar_detectores.py
    python comparar_detectores.py --criterio f1
    python comparar_detectores.py --criterio recall --max-falsos 15
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
                                fusionar_canal, coincidencia_onsets,
                                metricas_onset)

BANDA = (1.0, 20.0)
TOL_REPORTE_S = 5.0

# Rejilla centrada en la region util identificada por el barrido previo.
REJILLA_STALTA = dict(
    lta_s=[30.0, 60.0],
    sta_s=[1.0, 2.0],
    thr_on=[2.5, 3.0, 4.0, 5.0],
    thr_off=[1.05, 1.2],
    fusion_s=[5.0],
    min_est=[1, 2, 3],
)


def barrer_stalta(a, catalogo, valida, horas, canales):
    """Barrido de STA/LTA con metricas de NIVEL 1 (deteccion del momento)."""
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from obspy.signal.filter import bandpass

    n = a.shape[-1]
    print("Filtrando canales ...", flush=True)
    filtrado = {}
    for ch in canales:
        partes = []
        for ini, fin in segmentos(n):
            x = np.asarray(a[ch, ini:fin], dtype=np.float64)
            partes.append((ini, None if np.allclose(x, 0)
                           else bandpass(x, BANDA[0], BANDA[1], df=FS,
                                         corners=4, zerophase=True)))
        filtrado[ch] = partes
        print(f"  canal {ch} listo", flush=True)

    filas = []
    for lta_s, sta_s in itertools.product(REJILLA_STALTA["lta_s"],
                                          REJILLA_STALTA["sta_s"]):
        if sta_s >= lta_s:
            continue
        print(f">>> STA/LTA sta={sta_s}s lta={lta_s}s", flush=True)
        cft = {}
        for ch in canales:
            trozos = []
            for ini, xf in filtrado[ch]:
                if xf is None:
                    trozos.append((ini, None))
                else:
                    c = classic_sta_lta(xf, int(sta_s * FS), int(lta_s * FS))
                    trozos.append((ini, np.nan_to_num(c, nan=0.0, posinf=0.0,
                                                      neginf=0.0)))
            cft[ch] = trozos

        for thr_on, thr_off, fus_s in itertools.product(
                REJILLA_STALTA["thr_on"], REJILLA_STALTA["thr_off"],
                REJILLA_STALTA["fusion_s"]):
            if thr_off >= thr_on:
                continue
            por_canal = {}
            for ch in canales:
                trigs = []
                for ini, c in cft[ch]:
                    if c is None:
                        continue
                    for t0, t1 in trigger_onset(c, thr_on, thr_off):
                        trigs.append((ini + int(t0), ini + int(t1)))
                ivs = fusionar_canal(trigs, int(fus_s * FS))
                por_canal[ch] = {"onsets": [s for s, _ in ivs]}

            for k in REJILLA_STALTA["min_est"]:
                onsets = coincidencia_onsets(por_canal, valida, k)
                if not onsets:
                    continue
                d = metricas_onset(catalogo, onsets, horas,
                                   [TOL_REPORTE_S])[f"tol_{TOL_REPORTE_S:g}s"]
                rec, pre = d["recall"], d["precision"]
                filas.append(dict(
                    modelo="stalta",
                    modo=f"sta{sta_s:g}_lta{lta_s:g}_on{thr_on:g}_off{thr_off:g}",
                    umbral=thr_on, min_est=k, n_onsets=len(onsets),
                    recall=rec, precision=pre,
                    f1=2 * rec * pre / (rec + pre) if rec + pre else 0.0,
                    falsos_h=d["onsets_falsos_por_hora"],
                    err_mae=d["error_mae_s"],
                    **{f"rec_{c}": d["por_clase"].get(c, {}).get(
                        "recall", float("nan")) for c in CLASES}))
    return pd.DataFrame(filas)


def frontera_pareto(df):
    """Puntos no dominados en (recall alto, precision alta)."""
    d = df.sort_values("recall", ascending=False).reset_index(drop=True)
    mejor, out = -1.0, []
    for _, r in d.iterrows():
        if r["precision"] > mejor:
            out.append(r)
            mejor = r["precision"]
    return pd.DataFrame(out)


def graficar(tabla, destino):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        print(f"  (sin figura: {ex})")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    estilos = {"stalta": ("o", "#444444"), "phasenet": ("s", "#1f77b4"),
               "eqtransformer": ("^", "#d62728")}
    for modelo, gr in tabla.groupby("modelo"):
        marca, color = estilos.get(modelo, ("x", "gray"))
        ax.scatter(gr.recall, gr.precision, s=14, alpha=0.28,
                   marker=marca, color=color)
        fr = frontera_pareto(gr).sort_values("recall")
        ax.plot(fr.recall, fr.precision, marker=marca, color=color,
                lw=2, ms=6, label=f"{modelo} (frontera)")

    ax.set_xlabel("Recall de onset (tolerancia ±5 s)")
    ax.set_ylabel("Precisión")
    ax.set_title("Nivel 1: detección del momento — NVChVC, 10 h continuas")
    ax.grid(alpha=0.3, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    print(f"  figura: {destino}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterio", default="f1", choices=["f1", "recall"])
    ap.add_argument("--max-falsos", type=float, default=15.0,
                    help="techo de falsos/h cuando --criterio recall")
    ap.add_argument("--canales", nargs="+", type=int, default=CANALES)
    ap.add_argument("--recalcular-stalta", action="store_true")
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

    print(f"Catalogo: {len(catalogo)} eventos | {horas:.2f} h validas\n")

    # --- STA/LTA nivel 1 (con cache) --------------------------------
    ruta_st = os.path.join(SALIDA, "barrido_stalta_nivel1.csv")
    if os.path.exists(ruta_st) and not args.recalcular_stalta:
        print(f"Usando cache: {ruta_st}")
        t_st = pd.read_csv(ruta_st)
    else:
        t_st = barrer_stalta(a, catalogo, valida, horas, args.canales)
        os.makedirs(SALIDA, exist_ok=True)
        t_st.to_csv(ruta_st, index=False)
        print(f"Guardado: {ruta_st}")

    # --- modelos profundos ------------------------------------------
    ruta_mod = os.path.join(SALIDA, "barrido_modelos.csv")
    if not os.path.exists(ruta_mod):
        print(f"\nFalta {ruta_mod}. Corre primero barrido_modelos.py")
        return
    t_mod = pd.read_csv(ruta_mod)

    tabla = pd.concat([t_st, t_mod], ignore_index=True)
    tabla.to_csv(os.path.join(SALIDA, "comparacion_nivel1.csv"), index=False)

    # --- frontera de Pareto -----------------------------------------
    fronteras = []
    for modelo, gr in tabla.groupby("modelo"):
        f = frontera_pareto(gr)
        f["modelo"] = modelo
        fronteras.append(f)
    fr = pd.concat(fronteras, ignore_index=True)
    fr.to_csv(os.path.join(SALIDA, "frontera_pareto.csv"), index=False)

    print("\n" + "=" * 86)
    print("  FRONTERA PRECISION-RECALL POR DETECTOR")
    print("=" * 86)
    for modelo, gr in fr.groupby("modelo"):
        print(f"\n  {modelo}")
        print(f"    {'recall':>8}{'precision':>11}{'falsos/h':>10}"
              f"{'err.MAE':>9}   config")
        for _, r in gr.sort_values("recall", ascending=False).iterrows():
            print(f"    {r.recall:>8.3f}{r.precision:>11.3f}"
                  f"{r.falsos_h:>10.2f}{r.err_mae:>9.2f}   "
                  f"{r.modo} k={int(r.min_est)}")

    # --- punto de operacion con criterio unico ----------------------
    print("\n" + "=" * 86)
    if args.criterio == "f1":
        print("  PUNTO DE OPERACION: mejor F1 (mismo criterio para los tres)")
        sel = tabla.sort_values("f1", ascending=False)
    else:
        print(f"  PUNTO DE OPERACION: mayor recall con falsos/h <= "
              f"{args.max_falsos:g}")
        sel = tabla[tabla.falsos_h <= args.max_falsos] \
            .sort_values("recall", ascending=False)
    print("=" * 86)

    resumen = []
    for modelo in ["stalta", "phasenet", "eqtransformer"]:
        gr = sel[sel.modelo == modelo]
        if len(gr):
            resumen.append(gr.iloc[0])
    if not resumen:
        print("  Ningun detector cumple el criterio.")
        return
    res = pd.DataFrame(resumen)

    cols = ["modelo", "recall", "precision", "f1", "falsos_h", "err_mae"]
    print(res[cols].to_string(index=False,
                              float_format=lambda v: f"{v:.3f}"))
    print("\n  Configuracion elegida por detector:")
    for _, r in res.iterrows():
        print(f"    {r.modelo:<14} {r.modo}  min_est={int(r.min_est)}")

    print(f"\n  Recall por clase en el punto elegido:")
    print(f"    {'detector':<15}" + "".join(f"{c:>8}" for c in CLASES))
    for _, r in res.iterrows():
        print(f"    {r.modelo:<15}"
              + "".join(f"{r.get('rec_' + c, float('nan')):>8.3f}"
                        for c in CLASES))

    graficar(tabla, os.path.join(SALIDA, "curva_pr_nivel1.png"))

    print(f"""
NOTA PARA LA TESIS
  La frontera de Pareto es el resultado defendible: muestra si un detector
  domina a otro en todo el rango o solo en parte. El punto unico depende
  del criterio, y el criterio es una decision del observatorio, no del
  metodo. Conviene reportar la frontera completa y ademas un punto elegido
  con una regla declarada explicitamente.

  Recordar tambien que la tolerancia de +-5 s no es arbitraria: el inicio
  catalogado lo marco un analista humano y en eventos emergentes (LP, TR)
  tiene incertidumbre propia, probablemente del orden del segundo.
""")


if __name__ == "__main__":
    main()
