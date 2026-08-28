#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
significancia.py
================

Ultimo pendiente de la etapa de deteccion. Responde dos preguntas que un
evaluador va a hacer y que hasta ahora no estaban contestadas:

  1. Las diferencias entre detectores, son reales o son ruido estadistico?
     Con 203 eventos, una diferencia de recall de 0,04 puede no significar
     nada. Se calculan intervalos de confianza por bootstrap y, sobre todo,
     el intervalo de la DIFERENCIA entre pares de detectores: si incluye el
     cero, no se puede afirmar que uno sea mejor que otro.

  2. Las conclusiones dependen del umbral de IoU = 0,30 que se eligio?
     Se recalculan las metricas del Nivel 2 con IoU entre 0,10 y 0,50. Si
     el orden entre detectores se mantiene, la conclusion es robusta; si
     se invierte, el umbral estaba haciendo el trabajo.

El bootstrap es pareado: se remuestrean los MISMOS indices de evento para
todos los detectores, que es lo correcto porque los tres se evaluaron
sobre el mismo catalogo.

Uso:
    python significancia.py
    python significancia.py --n-boot 5000
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_detectores import (FS, CANALES, CORTES, GUARDA_S, NPY, CSV,
                                SALIDA, CLASES, Intervalo, segmentos,
                                binaria_a_intervalos, fusionar_canal,
                                coincidencia, coincidencia_onsets,
                                emparejar, metricas)

BANDA = (1.0, 20.0)
TOL_ONSET_S = 5.0
IOUS = [0.10, 0.20, 0.30, 0.40, 0.50]

# Puntos de operacion elegidos por "mejor F1", el mismo criterio para los
# tres. Salen de comparacion_nivel1.csv y barrido_eqt_nivel2.csv.
OP_NIVEL1 = {
    "stalta": dict(sta_s=2.0, lta_s=30.0, thr_on=5.0, thr_off=1.05,
                   fusion_s=5.0, min_est=2),
    "phasenet": dict(modo="ceros", umbral=0.05, min_est=2),
    "eqtransformer": dict(modo="ceros", umbral=0.30, min_est=1),
}
OP_NIVEL2 = {
    "stalta": dict(sta_s=2.0, lta_s=60.0, thr_on=4.0, thr_off=1.05,
                   fusion_s=5.0, min_est=2),
    "eqtransformer": dict(modo="ceros", umbral=0.05, fusion_s=15.0,
                          min_est=2),
}


# ---------------------------------------------------------------------
# Reconstruccion de las salidas de cada detector
# ---------------------------------------------------------------------

def stalta_por_canal(a, canales, cfg):
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from obspy.signal.filter import bandpass

    n = a.shape[-1]
    salida = {}
    for ch in canales:
        trigs = []
        for ini, fin in segmentos(n):
            x = np.asarray(a[ch, ini:fin], dtype=np.float64)
            if np.allclose(x, 0):
                continue
            xf = bandpass(x, BANDA[0], BANDA[1], df=FS, corners=4,
                          zerophase=True)
            cft = classic_sta_lta(xf, int(cfg["sta_s"] * FS),
                                  int(cfg["lta_s"] * FS))
            cft = np.nan_to_num(cft, nan=0.0, posinf=0.0, neginf=0.0)
            for t0, t1 in trigger_onset(cft, cfg["thr_on"], cfg["thr_off"]):
                trigs.append((ini + int(t0), ini + int(t1)))
        ivs = fusionar_canal(trigs, int(cfg["fusion_s"] * FS))
        salida[ch] = {"intervalos": ivs, "onsets": [s for s, _ in ivs]}
    return salida


def picks_por_canal(nombre, modo, umbral):
    ruta = os.path.join(SALIDA, "cache_picks", f"picks_{nombre}_{modo}.csv")
    if not os.path.exists(ruta):
        return None
    df = pd.read_csv(ruta)
    sel = df[df.conf >= umbral]
    return {ch: {"onsets": sorted(g.muestra.tolist())}
            for ch, g in sel.groupby("canal")}


def eqt_intervalos(modo, umbral, fusion_s, canales):
    ruta = os.path.join(SALIDA, "cache_detection", f"detection_eqt_{modo}.npz")
    if not os.path.exists(ruta):
        return None
    z = np.load(ruta)
    salida = {}
    for ch in canales:
        if str(ch) not in z.files:
            continue
        curva = z[str(ch)]
        ivs = [(iv.ini, iv.fin)
               for iv in binaria_a_intervalos(curva >= umbral)]
        salida[ch] = {"intervalos": fusionar_canal(ivs, int(fusion_s * FS))}
    return salida


# ---------------------------------------------------------------------
# Emparejamiento -> vector booleano por evento
# ---------------------------------------------------------------------

def acertados_onset(catalogo, onsets, tol_s=TOL_ONSET_S):
    """Devuelve (vector booleano por evento, n_onsets, n_acertados)."""
    onsets = np.asarray(sorted(onsets), dtype=np.int64)
    tol = int(tol_s * FS)
    cand = []
    for i, ev in enumerate(catalogo):
        for j, t in enumerate(onsets):
            d = abs(int(t) - ev.ini)
            if d <= tol:
                cand.append((d, i, j))
    cand.sort()
    ev_us, on_us = set(), set()
    for d, i, j in cand:
        if i in ev_us or j in on_us:
            continue
        ev_us.add(i)
        on_us.add(j)
    v = np.array([i in ev_us for i in range(len(catalogo))])
    return v, int(onsets.size), len(ev_us)


def acertados_iou(catalogo, detecciones, iou_min):
    pares, _, _ = emparejar(catalogo, detecciones, iou_min)
    ids = {id(g) for g, _, _ in pares}
    v = np.array([id(ev) in ids for ev in catalogo])
    return v, len(detecciones), len(pares)


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

def ic_bootstrap(v, n_boot, rng):
    """IC 95% de la proporcion de aciertos, remuestreando eventos."""
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    props = v[idx].mean(axis=1)
    return float(np.percentile(props, 2.5)), float(np.percentile(props, 97.5))


def ic_diferencia(v_a, v_b, n_boot, rng):
    """IC 95% de la diferencia de recall, con remuestreo PAREADO."""
    n = len(v_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = v_a[idx].mean(axis=1) - v_b[idx].mean(axis=1)
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return float(lo), float(hi), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canales", nargs="+", type=int, default=CANALES)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(20250101)

    a = np.load(NPY, mmap_mode="r")
    df = pd.read_csv(CSV)
    n = a.shape[-1]
    valida = np.ones(n, dtype=bool)
    g = int(GUARDA_S * FS)
    for c in CORTES:
        valida[max(0, c - g):min(n, c + g)] = False
    catalogo = [Intervalo(int(r.idx_start), int(r.idx_end), r.event_type)
                for r in df.itertuples()]
    catalogo = [e for e in catalogo if valida[e.ini] and valida[e.fin - 1]]
    horas = float(valida.sum()) / FS / 3600.0

    print(f"Catalogo: {len(catalogo)} eventos | {horas:.2f} h")
    print(f"Bootstrap: {args.n_boot} remuestreos pareados\n")

    # ---------- NIVEL 1 ----------
    print("Reconstruyendo salidas del Nivel 1 ...", flush=True)
    aciertos, resumen = {}, []

    pc = stalta_por_canal(a, args.canales, OP_NIVEL1["stalta"])
    ons = coincidencia_onsets(pc, valida, OP_NIVEL1["stalta"]["min_est"])
    v, n_det, n_tp = acertados_onset(catalogo, ons)
    aciertos["stalta"] = v
    resumen.append(("stalta", v, n_det, n_tp))
    print("  stalta listo", flush=True)

    for nombre in ["phasenet", "eqtransformer"]:
        cfg = OP_NIVEL1[nombre]
        pc = picks_por_canal(nombre, cfg["modo"], cfg["umbral"])
        if pc is None:
            print(f"  {nombre}: sin cache de picks, se omite")
            continue
        ons = coincidencia_onsets(pc, valida, cfg["min_est"])
        v, n_det, n_tp = acertados_onset(catalogo, ons)
        aciertos[nombre] = v
        resumen.append((nombre, v, n_det, n_tp))
        print(f"  {nombre} listo", flush=True)

    print("\n" + "=" * 78)
    print("  NIVEL 1 - RECALL CON INTERVALO DE CONFIANZA DEL 95 %")
    print("=" * 78)
    print(f"  {'detector':<16}{'recall':>9}{'IC 95%':>20}"
          f"{'precision':>11}{'n_onsets':>10}")
    for nombre, v, n_det, n_tp in resumen:
        rec = v.mean()
        lo, hi = ic_bootstrap(v, args.n_boot, rng)
        pre = n_tp / n_det if n_det else float("nan")
        print(f"  {nombre:<16}{rec:>9.3f}   [{lo:.3f} , {hi:.3f}]"
              f"{pre:>11.3f}{n_det:>10}")

    print("\n" + "=" * 78)
    print("  DIFERENCIAS ENTRE DETECTORES (bootstrap pareado)")
    print("=" * 78)
    print("  Si el intervalo incluye el cero, la diferencia NO es")
    print("  estadisticamente distinguible del ruido.\n")
    nombres = [r[0] for r in resumen]
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            na, nb = nombres[i], nombres[j]
            lo, hi, m = ic_diferencia(aciertos[na], aciertos[nb],
                                      args.n_boot, rng)
            signif = "NO" if lo <= 0 <= hi else "si"
            print(f"  {na:<14} - {nb:<14} = {m:+.3f}  "
                  f"IC [{lo:+.3f} , {hi:+.3f}]   significativa: {signif}")

    # ---------- NIVEL 1 POR CLASE ----------
    clases_ev = np.array([ev.clase for ev in catalogo])

    print("\n" + "=" * 78)
    print("  NIVEL 1 POR CLASE - RECALL CON IC 95 %")
    print("=" * 78)
    print("  Cada clase se remuestrea por separado, asi que el intervalo")
    print("  refleja el tamano de esa clase y no el del catalogo completo.\n")

    encabezado = f"  {'clase':<6}{'n':>4}"
    for nombre, *_ in resumen:
        encabezado += f"{nombre[:12]:>24}"
    print(encabezado)

    filas_clase = []
    for c in CLASES:
        mask = clases_ev == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        linea = f"  {c:<6}{n_c:>4}"
        for nombre, *_ in resumen:
            v_c = aciertos[nombre][mask]
            rec = v_c.mean()
            lo, hi = ic_bootstrap(v_c, args.n_boot, rng)
            linea += f"   {rec:.3f} [{lo:.2f},{hi:.2f}]"
            filas_clase.append(dict(nivel=1, clase=c, n=n_c, detector=nombre,
                                    recall=float(rec), ic_lo=lo, ic_hi=hi))
        print(linea)

    print("\n" + "=" * 78)
    print("  DIFERENCIAS POR CLASE (bootstrap pareado dentro de cada clase)")
    print("=" * 78)
    print(f"  {'clase':<6}{'n':>4}  {'comparacion':<32}{'dif':>8}"
          f"{'IC 95%':>20}{'signif':>9}")
    n_signif = 0
    for c in CLASES:
        mask = clases_ev == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        for i in range(len(nombres)):
            for j in range(i + 1, len(nombres)):
                na, nb = nombres[i], nombres[j]
                lo, hi, m = ic_diferencia(aciertos[na][mask],
                                          aciertos[nb][mask],
                                          args.n_boot, rng)
                sig = not (lo <= 0 <= hi)
                n_signif += int(sig)
                print(f"  {c:<6}{n_c:>4}  {na + ' - ' + nb:<32}{m:>+8.3f}"
                      f"   [{lo:+.3f} , {hi:+.3f}]{'si' if sig else 'NO':>9}")
    print(f"\n  Comparaciones significativas: {n_signif} de "
          f"{len(CLASES) * len(nombres) * (len(nombres) - 1) // 2}")

    # ---------- NIVEL 2: sensibilidad al IoU ----------
    print("\n" + "=" * 78)
    print("  NIVEL 2 - SENSIBILIDAD AL UMBRAL DE IoU")
    print("=" * 78)

    print("\nReconstruyendo salidas del Nivel 2 ...", flush=True)
    det2 = {}
    cfg = OP_NIVEL2["stalta"]
    pc = stalta_por_canal(a, args.canales, cfg)
    det2["stalta"] = coincidencia(pc, valida, cfg["min_est"])
    print("  stalta listo", flush=True)

    cfg = OP_NIVEL2["eqtransformer"]
    pc = eqt_intervalos(cfg["modo"], cfg["umbral"], cfg["fusion_s"],
                        args.canales)
    if pc is not None:
        det2["eqtransformer"] = coincidencia(pc, valida, cfg["min_est"])
        print("  eqtransformer listo", flush=True)

    filas = []
    for iou_min in IOUS:
        print(f"\n  --- IoU minimo = {iou_min:.2f} ---")
        print(f"  {'detector':<16}{'recall':>9}{'IC 95%':>20}"
              f"{'precision':>11}{'F1':>8}{'IoU medio':>11}")
        vecs = {}
        for nombre, det in det2.items():
            v, n_d, n_tp = acertados_iou(catalogo, det, iou_min)
            vecs[nombre] = v
            r, *_ = metricas(catalogo, det, horas, iou_min)
            lo, hi = ic_bootstrap(v, args.n_boot, rng)
            print(f"  {nombre:<16}{r['recall']:>9.3f}   "
                  f"[{lo:.3f} , {hi:.3f}]{r['precision']:>11.3f}"
                  f"{r['f1']:>8.3f}{r['iou_medio']:>11.3f}")
            filas.append(dict(iou_min=iou_min, detector=nombre,
                              recall=r["recall"], ic_lo=lo, ic_hi=hi,
                              precision=r["precision"], f1=r["f1"],
                              iou_medio=r["iou_medio"],
                              fp_h=r["fp_por_hora"]))
        if len(vecs) == 2:
            (na, va), (nb, vb) = list(vecs.items())
            lo, hi, m = ic_diferencia(va, vb, args.n_boot, rng)
            signif = "NO" if lo <= 0 <= hi else "si"
            print(f"    diferencia {na} - {nb} = {m:+.3f}  "
                  f"IC [{lo:+.3f} , {hi:+.3f}]  significativa: {signif}")

    if filas or filas_clase:
        os.makedirs(SALIDA, exist_ok=True)
        destino = os.path.join(SALIDA, "significancia.csv")
        pd.DataFrame(filas).to_csv(destino, index=False)
        if filas_clase:
            destino_c = os.path.join(SALIDA, "significancia_por_clase.csv")
            pd.DataFrame(filas_clase).to_csv(destino_c, index=False)
            print(f"\nTabla por clase en {destino_c}")
        print(f"Tabla en {destino}")

    print("""
COMO LEER ESTO
  Nivel 1: si los intervalos de confianza de dos detectores se solapan
  ampliamente y el intervalo de su diferencia incluye el cero, no se
  puede afirmar que uno supere al otro. En ese caso la conclusion
  correcta es que son indistinguibles en recall, y la comparacion debe
  apoyarse en las metricas donde si hay separacion (error de onset,
  techo de recall alcanzable, desempeno por clase).

  Por clase: los intervalos son necesariamente mas anchos porque cada
  clase tiene entre 23 y 55 eventos. Una diferencia que aparece grande
  en la tabla de recall puede no ser distinguible del ruido. Solo las
  comparaciones marcadas como significativas admiten afirmarse.

  Nivel 2: si el orden entre detectores se mantiene en todo el rango de
  IoU, la conclusion es robusta y no depende del umbral elegido. Si se
  invierte en algun punto, hay que declararlo y explicar por que se fijo
  el umbral en 0,30.
""")


if __name__ == "__main__":
    main()
