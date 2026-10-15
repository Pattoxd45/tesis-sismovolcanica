#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corre PhaseNet y EQTransformer sobre FRE y guarda las curvas de probabilidad.

Uso: python anotar_fre3c.py [--modelos ...] [--filtros ...] [--sin-verificar]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fre3c_lib as lib

FS = lib.FS
PESOS = "original"
UMBRAL_V4 = 0.3
TRAMO_V4 = 2


def cargar_modelo(nombre):
    import seisbench.models as sbm
    clase = {"phasenet": sbm.PhaseNet, "eqtransformer": sbm.EQTransformer}[nombre]
    m = clase.from_pretrained(PESOS)
    m.eval()
    return m


def configuracion(m, nombre):
    import seisbench
    def plano(v):
        try:
            json.dumps(v)
            return v
        except TypeError:
            return repr(v)
    return dict(
        modelo=nombre, pesos=PESOS, seisbench=seisbench.__version__,
        clase=m.__class__.__name__,
        component_order=plano(getattr(m, "component_order", None)),
        sampling_rate=plano(getattr(m, "sampling_rate", None)),
        in_samples=plano(getattr(m, "in_samples", None)),
        norm=plano(getattr(m, "norm", None)),
        labels=plano(getattr(m, "labels", None)),
        filter_args=plano(getattr(m, "filter_args", None)),
        filter_kwargs=plano(getattr(m, "filter_kwargs", None)),
        default_args=plano(getattr(m, "default_args", None)),
        annotate_args={k: plano(v[1]) for k, v in getattr(m, "_annotate_args", {}).items()},
    )


def stream_tramo(datos, a, b):
    from obspy import Stream, Trace, UTCDateTime
    trs = []
    for c in ("Z", "N", "E"):
        tr = Trace(data=np.ascontiguousarray(datos[c][a:b], dtype=np.float32))
        tr.stats.sampling_rate = FS
        tr.stats.network = "NV"
        tr.stats.station = "FRE"
        tr.stats.channel = "HH" + c
        tr.stats.starttime = UTCDateTime(0) + a / FS
        trs.append(tr)
    return Stream(trs)


def volcar(ann, curvas, a, b):
    for tr in ann:
        etiqueta = tr.stats.channel.split("_")[-1]
        if abs(tr.stats.sampling_rate - FS) > 1e-6:
            raise SystemExit(f"DETENIDO: la anotacion {tr.id} viene a "
                             f"{tr.stats.sampling_rate} Hz, se esperaban {FS} Hz.")
        off = int(round(tr.stats.starttime.timestamp * FS))
        lo, hi = max(a, off), min(b, off + tr.stats.npts)
        if hi <= lo:
            continue
        if etiqueta not in curvas:
            curvas[etiqueta] = np.full(lib.N_TOTAL, np.nan, dtype=np.float32)
        curvas[etiqueta][lo:hi] = tr.data[lo - off:hi - off]


def verificar_v4(m, datos, curvas, nombre):
    """Compara los disparos propios con classify() de SeisBench en el tramo 2."""
    a, b = lib.LIMITES_TRAMO[TRAMO_V4 - 1]
    st = stream_tramo(datos, a, b)
    kw = dict(P_threshold=UMBRAL_V4, S_threshold=UMBRAL_V4)
    if nombre == "eqtransformer":
        kw["detection_threshold"] = UMBRAL_V4
    out = m.classify(st, **kw)
    ok = True
    for fase in ("P", "S"):
        sb = sorted(int(round(p.peak_time.timestamp * FS)) for p in out.picks
                    if str(p.phase) == fase)
        _, _, pico, _ = lib.disparos(curvas[fase][a:b], UMBRAL_V4)
        propio = sorted((pico + a).tolist())
        igual = len(sb) == len(propio) and all(abs(x - y) <= 1 for x, y in zip(sb, propio))
        ok &= igual
        print(f"    V4 {nombre} {fase}: classify={len(sb)} propios={len(propio)} "
              f"-> {'IGUALES' if igual else 'DISTINTOS'}")
    if nombre == "eqtransformer":
        sb = sorted(int(round(d.start_time.timestamp * FS)) for d in out.detections)
        ini, _, _, _ = lib.disparos(curvas["Detection"][a:b], UMBRAL_V4)
        propio = sorted((ini + a).tolist())
        igual = len(sb) == len(propio) and all(abs(x - y) <= 1 for x, y in zip(sb, propio))
        ok &= igual
        print(f"    V4 {nombre} Detection: classify={len(sb)} propios={len(propio)} "
              f"-> {'IGUALES' if igual else 'DISTINTOS'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["phasenet", "eqtransformer"],
                    choices=["phasenet", "eqtransformer"])
    ap.add_argument("--filtros", nargs="+", default=list(lib.FILTROS),
                    choices=list(lib.FILTROS))
    ap.add_argument("--sin-verificar", action="store_true")
    args = ap.parse_args()

    v4_global = True
    for nombre in args.modelos:
        m = cargar_modelo(nombre)
        cfg = configuracion(m, nombre)
        print(f"\n=== {nombre}: configuracion cargada por SeisBench ===")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        if cfg["filter_args"] not in (None, "None") or cfg["filter_kwargs"] not in (None, "None"):
            print("  ATENCION: el modelo aplica un filtro propio antes de inferir.")
        else:
            print("  El modelo NO aplica filtro propio: recibe la senal tal como se le entrega.")

        for filtro in args.filtros:
            ruta = lib.ruta_datos(filtro)
            if not os.path.exists(ruta):
                raise SystemExit(f"Falta {ruta}: correr antes preparar_fre3c.py")
            z = np.load(ruta)
            datos = {c: z[c] for c in ("Z", "N", "E")}
            curvas = {}
            t0 = time.time()
            for k, (a, b) in enumerate(lib.LIMITES_TRAMO, 1):
                print(f"  {nombre}/{filtro} tramo {k} ...", flush=True)
                ann = m.annotate(stream_tramo(datos, a, b))
                volcar(ann, curvas, a, b)
            seg = time.time() - t0
            print(f"  etiquetas: {sorted(curvas)} | {seg:.0f} s de computo")
            for et, cv in curvas.items():
                fin = np.isfinite(cv)
                print(f"    {et}: anotado {fin.mean():.4f} de la traza, "
                      f"max {np.nanmax(cv):.3f}, p99 {np.nanpercentile(cv, 99):.3f}")

            if not args.sin_verificar:
                v4_global &= verificar_v4(m, datos, curvas, nombre)

            np.savez_compressed(lib.ruta_anot(nombre, filtro), **curvas)
            with open(lib.ruta_anot(nombre, filtro).replace(".npz", ".json"), "w") as f:
                json.dump(dict(cfg, filtro=filtro, banda=list(lib.FILTROS[filtro]),
                               segundos=seg), f, indent=2, ensure_ascii=False)
            print(f"  guardado {lib.ruta_anot(nombre, filtro)}")

    if not args.sin_verificar:
        print(f"\nV4 global: {'PASA' if v4_global else 'FALLA (revisar antes de usar la tabla)'}")


if __name__ == "__main__":
    main()
