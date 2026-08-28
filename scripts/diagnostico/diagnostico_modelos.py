#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostico_modelos.py
======================

Inspecciona QUE devuelven realmente PhaseNet y EQTransformer sobre un
evento conocido del catalogo, antes de rediseñar la evaluacion.

Preguntas que responde:
  1. Que canales entrega annotate() y como se llaman?
  2. Que forma tienen las curvas: picos agudos o mesetas?
  3. Cuantas muestras superan cada umbral? (mide si son intervalos o picos)
  4. classify() entrega picks utiles? cuantos y donde?
  5. Cambia algo al duplicar la vertical en las tres componentes?
  6. Queda bien alineada la salida con la entrada?

Es rapido: trabaja sobre ventanas de pocos minutos, no sobre las 10 h.

Uso:
    python diagnostico_modelos.py
    python diagnostico_modelos.py --modelos phasenet
    python diagnostico_modelos.py --n-eventos 5
"""

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = "/home/patto/tesis/datos/zenodo"
NPY = f"{BASE}/NVCh_10h_continuous_trace.npy"
CSV = f"{BASE}/NVCh_10h_continuous_trace_reference.csv"

FS = 100.0
CANAL = 1          # canal con mejor relacion senal/ruido (sigma mas alta)
MARGEN_S = 60.0    # contexto a cada lado del evento


def construir_stream(x, ini_muestra, modo, estacion="S1"):
    from obspy import Trace, Stream, UTCDateTime
    t0 = UTCDateTime(0) + ini_muestra / FS
    comps = ["Z", "N", "E"] if modo == "duplicar" else ["Z"]
    trs = []
    for c in comps:
        tr = Trace(data=np.ascontiguousarray(x, dtype=np.float32))
        tr.stats.sampling_rate = FS
        tr.stats.network = "NV"
        tr.stats.station = estacion
        tr.stats.channel = "HH" + c
        tr.stats.starttime = t0
        trs.append(tr)
    return Stream(trs)


def analizar(modelo, nombre, x, ini, s_rel, e_rel, modo):
    """Corre el modelo sobre una ventana y describe su salida."""
    st = construir_stream(x, ini, modo)

    print(f"\n  --- {nombre} | componentes={modo} ---")
    try:
        ann = modelo.annotate(st)
    except Exception as ex:
        print(f"    annotate() fallo: {type(ex).__name__}: {ex}")
        return

    if len(ann) == 0:
        print("    annotate() devolvio un Stream VACIO")
        return

    print(f"    canales devueltos: {[tr.stats.channel for tr in ann]}")
    off = int(round(ann[0].stats.starttime.timestamp * FS)) - ini
    print(f"    desfase de la salida: {off} muestras ({off / FS:.2f} s) | "
          f"npts entrada={len(x)} salida={ann[0].stats.npts}")

    for tr in ann:
        d = np.asarray(tr.data, dtype=np.float64)
        if d.size == 0:
            continue
        # region del evento dentro de la salida anotada
        a0 = max(0, s_rel - off)
        a1 = min(d.size, e_rel - off)
        dentro = d[a0:a1] if a1 > a0 else np.array([0.0])

        sup = {u: int((d >= u).sum()) for u in (0.1, 0.3, 0.5)}
        # ancho tipico de las regiones que superan 0,3
        m = (d >= 0.3).astype(np.int8)
        dd = np.diff(np.concatenate(([0], m, [0])))
        anchos = np.where(dd == -1)[0] - np.where(dd == 1)[0]
        ancho_med = float(np.median(anchos)) / FS if anchos.size else 0.0

        print(f"      {tr.stats.channel:<24} "
              f"max={d.max():.3f} media={d.mean():.3f} | "
              f"en evento: max={dentro.max():.3f} media={dentro.mean():.3f}")
        print(f"      {'':<24} muestras>=0.1:{sup[0.1]:>6} "
              f">=0.3:{sup[0.3]:>6} >=0.5:{sup[0.5]:>6} | "
              f"ancho mediano de region>=0.3: {ancho_med:.2f} s")

    # picks explicitos
    try:
        out = modelo.classify(st)
        picks = getattr(out, "picks", out)
        print(f"    classify(): {len(picks)} picks")
        for p in list(picks)[:6]:
            t = getattr(p, "peak_time", None)
            idx = (int(round(t.timestamp * FS)) - ini) if t is not None else -1
            print(f"      fase={getattr(p, 'phase', '?')} "
                  f"muestra_rel={idx} (evento en {s_rel}..{e_rel}) "
                  f"conf={getattr(p, 'peak_value', float('nan')):.3f}")
        det = getattr(out, "detections", None)
        if det is not None:
            print(f"    classify(): {len(det)} detecciones de intervalo")
            for dd_ in list(det)[:4]:
                si = int(round(dd_.start_time.timestamp * FS)) - ini
                ei = int(round(dd_.end_time.timestamp * FS)) - ini
                print(f"      intervalo rel {si}..{ei} "
                      f"({(ei - si) / FS:.1f} s)")
    except Exception as ex:
        print(f"    classify() fallo: {type(ex).__name__}: {ex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+",
                    default=["phasenet", "eqtransformer"])
    ap.add_argument("--n-eventos", type=int, default=3)
    ap.add_argument("--canal", type=int, default=CANAL)
    args = ap.parse_args()

    import seisbench.models as sbm

    a = np.load(NPY, mmap_mode="r")
    df = pd.read_csv(CSV)

    # eventos de referencia: uno corto, uno medio, uno largo
    df = df.assign(dur=(df.idx_end - df.idx_start) / FS)
    elegidos = []
    for clase in ["VT", "LP", "TR"]:
        sub = df[df.event_type == clase].sort_values("dur")
        if len(sub):
            elegidos.append(sub.iloc[len(sub) // 2])
    elegidos = elegidos[:args.n_eventos]

    modelos = {}
    if "phasenet" in args.modelos:
        modelos["PhaseNet"] = sbm.PhaseNet.from_pretrained("original")
    if "eqtransformer" in args.modelos:
        modelos["EQTransformer"] = sbm.EQTransformer.from_pretrained("original")
    for m in modelos.values():
        m.eval()

    marg = int(MARGEN_S * FS)
    for ev in elegidos:
        s, e = int(ev.idx_start), int(ev.idx_end)
        ini = max(0, s - marg)
        fin = min(a.shape[-1], e + marg)
        x = np.asarray(a[args.canal, ini:fin], dtype=np.float64)

        print("\n" + "=" * 72)
        print(f"  EVENTO {ev.event_type} | muestras {s}..{e} "
              f"| duracion {ev.dur:.1f} s | canal {args.canal}")
        print(f"  ventana analizada: {ini}..{fin} "
              f"({(fin - ini) / FS:.0f} s) | amplitud pico={np.abs(x).max():.0f}")
        print("=" * 72)

        for nombre, modelo in modelos.items():
            for modo in ["ceros", "duplicar"]:
                analizar(modelo, nombre, x, ini, s - ini, e - ini, modo)

    print("\n" + "=" * 72)
    print("  COMO LEER ESTO")
    print("=" * 72)
    print("""
  - 'ancho mediano de region>=0.3' cercano a 0.0-0.5 s -> la salida son
    PICOS (arribos), no intervalos: no sirve umbralizarla para recortar.
  - Si EQTransformer devuelve un canal 'Detection' con anchos de decenas
    de segundos -> ese si delimita eventos y sirve para el IoU.
  - Si 'duplicar' cambia mucho los numeros -> el relleno de componentes
    importa y hay que reportarlo como decision metodologica.
  - Si classify() devuelve picks dentro del rango del evento, PhaseNet
    SI esta detectando el momento, aunque no delimite el fin.
""")


if __name__ == "__main__":
    main()
