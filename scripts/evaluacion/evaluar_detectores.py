#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluar_detectores.py  (v2)
===========================

Evaluacion comparativa de detectores de eventos sismo-volcanicos sobre la
traza continua de 10 h del Complejo Volcanico Nevados de Chillan.

Datos: Zenodo DOI 10.5281/zenodo.17163020 (v4)

EVALUACION EN DOS NIVELES
-------------------------
El diagnostico empirico mostro que los detectores no producen la misma
clase de salida, asi que compararlos con una sola metrica es injusto:

    STA/LTA        -> intervalos (inicio y fin)
    EQTransformer  -> intervalos (canal Detection) + picks P/S
    PhaseNet       -> solo picks P/S, no delimita el fin del evento

NIVEL 1 - DETECCION DEL MOMENTO (los tres compiten en igualdad)
    Se pregunta si el detector marco un inicio dentro de +-T segundos del
    inicio catalogado, para T = 1, 2 y 5 s. Metricas: recall de onset,
    onsets falsos por hora, sesgo y dispersion del error de onset.

NIVEL 2 - CALIDAD DEL RECORTE (solo detectores que delimitan)
    IoU, error de offset y sesgo de duracion. PhaseNet queda fuera por
    construccion, y eso mismo es un resultado.

RELLENO DE COMPONENTES
----------------------
Los datos son de UNA componente (vertical) por estacion, pero los modelos
esperan tripletas Z-N-E. Medido sobre eventos de prueba:

    PhaseNet      : con ceros queda CIEGO (P max 0,108, cero picks).
                    Con la vertical duplicada, P max 0,910. -> "duplicar"
    EQTransformer : con ceros Detection llega a 0,98; duplicando cae
                    a 0,36. -> "ceros"

La estrategia optima depende del modelo. Por eso el modo se fija por
detector y se puede forzar con --componentes para el analisis de
sensibilidad.

Uso:
    python evaluar_detectores.py --detectores stalta
    python evaluar_detectores.py --detectores stalta phasenet eqtransformer
    python evaluar_detectores.py --detectores phasenet --componentes ceros
    python evaluar_detectores.py --autotest

Autor: Patricio Valdes Troncoso - Universidad Catolica de Temuco
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURACION
# =====================================================================

BASE = "/home/patto/tesis/datos/zenodo"
NPY = f"{BASE}/NVCh_10h_continuous_trace.npy"
CSV = f"{BASE}/NVCh_10h_continuous_trace_reference.csv"
SALIDA = "/home/patto/tesis/resultados"

FS = 100.0  # Hz

# Estaciones utilizables. Se excluye la 8 (identicamente cero), la 7
# (70,8% sin dato) y la 4 (sigma ~100x menor: ganancia anomala).
# Debe ser el MISMO conjunto para los tres detectores.
CANALES = [1, 2, 3, 5, 6]

# La traza concatena tres segmentos de fechas distintas (saltos de anos).
CORTES = [2_160_000, 3_240_000]
GUARDA_S = 30.0

# --- Nivel 1: deteccion del momento --------------------------------
TOL_ONSET_S = [1.0, 2.0, 5.0]   # tolerancias reportadas
TOL_RED_ONSET_S = 3.0           # agrupacion de onsets entre estaciones

# --- Nivel 2: calidad del recorte ----------------------------------
IOU_MIN = 0.30

# --- Agregacion a nivel de red -------------------------------------
MIN_ESTACIONES = 2
TOL_COINCIDENCIA_S = 5.0
FUSION_S = 5.0
DUR_MIN_S = 3.0
# Tope de duracion de una deteccion de red. El evento mas largo del
# catalogo dura 240,8 s (clase TR); 400 s deja margen holgado y corta el
# encadenamiento transitivo que producia "detecciones" de mas de una hora.
DUR_MAX_S = 400.0

# --- STA/LTA (optimo del barrido de 324 combinaciones) -------------
STALTA = dict(sta_s=2.0, lta_s=60.0, thr_on=4.0, thr_off=1.05,
              banda=(1.0, 20.0))

# --- Modelos SeisBench ---------------------------------------------
PESOS = {"phasenet": "original", "eqtransformer": "original"}
COMPONENTES_POR_MODELO = {"phasenet": "duplicar", "eqtransformer": "ceros"}
UMBRAL_PROB = 0.30   # sobre la curva Detection (solo EQTransformer)
UMBRAL_PICK = 0.20   # confianza minima de un pick

CLASES = ["VT", "LP", "TR", "AV", "IC"]


# =====================================================================
# NUCLEO (testeable sin SeisBench)
# =====================================================================

@dataclass
class Intervalo:
    ini: int
    fin: int
    clase: str = ""
    score: float = 1.0

    @property
    def dur(self) -> int:
        return self.fin - self.ini


def iou(a: Intervalo, b: Intervalo) -> float:
    inter = max(0, min(a.fin, b.fin) - max(a.ini, b.ini))
    if inter == 0:
        return 0.0
    union = max(a.fin, b.fin) - min(a.ini, b.ini)
    return inter / union if union > 0 else 0.0


def binaria_a_intervalos(mask, dur_min=0):
    mask = np.asarray(mask).astype(np.int8)
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask, [0])))
    inis = np.where(d == 1)[0]
    fins = np.where(d == -1)[0]
    return [Intervalo(int(i), int(f)) for i, f in zip(inis, fins)
            if (f - i) >= dur_min]


def fusionar_canal(trigs, gap):
    """Une disparos del MISMO canal separados por menos de gap muestras."""
    if not trigs:
        return []
    trigs = sorted(trigs)
    out = [list(trigs[0])]
    for s, e in trigs[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def coincidencia(por_canal, valida, min_est=MIN_ESTACIONES,
                 tol_s=TOL_COINCIDENCIA_S, dur_min_s=DUR_MIN_S,
                 dur_max_s=DUR_MAX_S):
    """
    Consolida intervalos de cada canal en detecciones de red.

    La coincidencia se evalua por INTERVALO con ventana de tolerancia y
    no muestra a muestra: los disparos duran pocos segundos y llegan
    desfasados entre estaciones. Exigir solapamiento exacto hundia el
    recall de 0,59 a 0,05.

    TOPE DE DURACION: el agrupamiento es transitivo (si A toca a B y B
    toca a C, los tres caen en el mismo grupo). En periodos sismicamente
    activos con varios canales la cadena se propaga y puede producir una
    unica "deteccion" de decenas de minutos, que ademas absorbe eventos
    reales y los convierte en falsos negativos. Cuando un grupo excede
    dur_max_s se corta en bloques, en el hueco interno mas ancho.
    """
    tol = int(tol_s * FS)
    dur_min = int(dur_min_s * FS)
    dur_max = int(dur_max_s * FS)

    todos = [(s, e, ch) for ch, d in por_canal.items()
             for s, e in d.get("intervalos", [])]
    if not todos:
        return []
    todos.sort()

    grupos, actual, fin_act = [], [todos[0]], todos[0][1]
    for s, e, ch in todos[1:]:
        # se corta si no hay contacto O si el grupo ya es demasiado largo
        if s <= fin_act + tol and (max(fin_act, e) - actual[0][0]) <= dur_max:
            actual.append((s, e, ch))
            fin_act = max(fin_act, e)
        else:
            grupos.append(actual)
            actual, fin_act = [(s, e, ch)], e
    grupos.append(actual)

    salida = []
    for g in grupos:
        if len({ch for _, _, ch in g}) < min_est:
            continue
        ini = int(min(s for s, _, _ in g))
        fin = int(max(e for _, e, _ in g))
        if dur_min <= fin - ini <= dur_max and valida[ini] and valida[fin - 1]:
            salida.append(Intervalo(ini, fin))
    return salida


def coincidencia_onsets(por_canal, valida, min_est=MIN_ESTACIONES,
                        tol_s=TOL_RED_ONSET_S):
    """
    Agrupa onsets (instantes) de distintas estaciones y emite un onset de
    red por grupo, usando la MEDIANA del grupo como estimador robusto.
    """
    tol = int(tol_s * FS)
    todos = sorted((int(t), ch) for ch, d in por_canal.items()
                   for t in d.get("onsets", []))
    if not todos:
        return []

    grupos, actual = [], [todos[0]]
    for t, ch in todos[1:]:
        if t - actual[-1][0] <= tol:
            actual.append((t, ch))
        else:
            grupos.append(actual)
            actual = [(t, ch)]
    grupos.append(actual)

    salida = []
    for g in grupos:
        if len({ch for _, ch in g}) < min_est:
            continue
        t = int(np.median([t for t, _ in g]))
        if 0 <= t < len(valida) and valida[t]:
            salida.append(t)
    return sorted(salida)


# ---------------------------------------------------------------------
# NIVEL 1: metricas de deteccion del momento
# ---------------------------------------------------------------------

def metricas_onset(catalogo, onsets, horas, tolerancias=TOL_ONSET_S):
    """
    Empareja cada evento del catalogo con el onset detectado mas cercano
    a su inicio, dentro de la tolerancia. Asignacion 1 a 1.
    """
    onsets = np.asarray(sorted(onsets), dtype=np.int64)
    res = {"n_catalogo": len(catalogo), "n_onsets": int(onsets.size)}

    for tol_s in tolerancias:
        tol = int(tol_s * FS)
        cand = []
        for i, ev in enumerate(catalogo):
            for j, t in enumerate(onsets):
                d = abs(int(t) - ev.ini)
                if d <= tol:
                    cand.append((d, i, j))
        cand.sort()

        ev_us, on_us, pares = set(), set(), []
        for d, i, j in cand:
            if i in ev_us or j in on_us:
                continue
            ev_us.add(i)
            on_us.add(j)
            pares.append((catalogo[i], int(onsets[j])))

        n_tp = len(pares)
        n_fp = int(onsets.size) - n_tp
        clave = f"tol_{tol_s:g}s"
        d_res = {
            "recall": n_tp / len(catalogo) if catalogo else float("nan"),
            "precision": n_tp / onsets.size if onsets.size else float("nan"),
            "onsets_falsos_por_hora": n_fp / horas if horas else float("nan"),
        }
        if pares:
            err = np.array([(t - ev.ini) / FS for ev, t in pares])
            d_res["error_mediano_s"] = float(np.median(err))
            d_res["error_mae_s"] = float(np.mean(np.abs(err)))
        else:
            d_res["error_mediano_s"] = float("nan")
            d_res["error_mae_s"] = float("nan")

        # desglose por clase
        d_res["por_clase"] = {}
        for c in CLASES:
            n_c = sum(1 for ev in catalogo if ev.clase == c)
            if n_c:
                tp_c = sum(1 for ev, _ in pares if ev.clase == c)
                d_res["por_clase"][c] = {"n": n_c, "recall": tp_c / n_c}
        res[clave] = d_res
    return res


# ---------------------------------------------------------------------
# NIVEL 2: metricas de calidad del recorte
# ---------------------------------------------------------------------

def emparejar(catalogo, detecciones, iou_min=IOU_MIN):
    cand = []
    for i, gt in enumerate(catalogo):
        for j, det in enumerate(detecciones):
            v = iou(gt, det)
            if v >= iou_min:
                cand.append((v, i, j))
    cand.sort(reverse=True)

    gt_us, det_us, pares = set(), set(), []
    for v, i, j in cand:
        if i in gt_us or j in det_us:
            continue
        gt_us.add(i)
        det_us.add(j)
        pares.append((catalogo[i], detecciones[j], v))

    perdidos = [g for i, g in enumerate(catalogo) if i not in gt_us]
    fp = [d for j, d in enumerate(detecciones) if j not in det_us]
    return pares, perdidos, fp


def metricas(catalogo, detecciones, horas_validas, iou_min=IOU_MIN):
    pares, perdidos, fp = emparejar(catalogo, detecciones, iou_min)
    n_gt, n_det, n_tp = len(catalogo), len(detecciones), len(pares)

    res = {
        "n_catalogo": n_gt,
        "n_detecciones": n_det,
        "verdaderos_positivos": n_tp,
        "falsos_positivos": len(fp),
        "no_detectados": len(perdidos),
        "recall": n_tp / n_gt if n_gt else float("nan"),
        "precision": n_tp / n_det if n_det else float("nan"),
        "fp_por_hora": len(fp) / horas_validas if horas_validas else float("nan"),
    }
    r, p = res["recall"], res["precision"]
    res["f1"] = 2 * r * p / (r + p) if (r + p) > 0 else 0.0

    claves = ["onset_mediana_s", "onset_mae_s", "onset_p90_s",
              "offset_mediana_s", "offset_mae_s", "duracion_sesgo_s",
              "iou_medio", "iou_mediano"]
    if pares:
        e_on = np.array([(d.ini - g.ini) / FS for g, d, _ in pares])
        e_off = np.array([(d.fin - g.fin) / FS for g, d, _ in pares])
        e_dur = np.array([(d.dur - g.dur) / FS for g, d, _ in pares])
        ious = np.array([v for _, _, v in pares])
        res.update({
            "onset_mediana_s": float(np.median(e_on)),
            "onset_mae_s": float(np.mean(np.abs(e_on))),
            "onset_p90_s": float(np.percentile(np.abs(e_on), 90)),
            "offset_mediana_s": float(np.median(e_off)),
            "offset_mae_s": float(np.mean(np.abs(e_off))),
            "duracion_sesgo_s": float(np.median(e_dur)),
            "iou_medio": float(np.mean(ious)),
            "iou_mediano": float(np.median(ious)),
        })
    else:
        for k in claves:
            res[k] = float("nan")

    por_clase = {}
    for c in CLASES:
        gt_c = [g for g in catalogo if g.clase == c]
        if not gt_c:
            continue
        tp_c = [(g, d, v) for g, d, v in pares if g.clase == c]
        d = {"n": len(gt_c), "recall": len(tp_c) / len(gt_c),
             "dur_mediana_s": float(np.median([g.dur for g in gt_c]) / FS)}
        if tp_c:
            d["onset_mae_s"] = float(np.mean(
                [abs(dd.ini - gg.ini) / FS for gg, dd, _ in tp_c]))
            d["iou_medio"] = float(np.mean([v for _, _, v in tp_c]))
        else:
            d["onset_mae_s"] = float("nan")
            d["iou_medio"] = float("nan")
        por_clase[c] = d
    res["por_clase"] = por_clase
    return res, pares, perdidos, fp


# =====================================================================
# CARGA
# =====================================================================

def cargar():
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
    return a, catalogo, valida, horas


def segmentos(n):
    b = [0] + CORTES + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


# =====================================================================
# DETECTORES
# =====================================================================

def _stream(datos, ini_global, modo, estacion="S1"):
    """Stream de ObsPy alineado a indices de muestra (starttime = ini/FS)."""
    from obspy import Trace, Stream, UTCDateTime

    t0 = UTCDateTime(0) + ini_global / FS
    comps = ["Z", "N", "E"] if modo == "duplicar" else ["Z"]
    trs = []
    for c in comps:
        tr = Trace(data=np.ascontiguousarray(datos, dtype=np.float32))
        tr.stats.sampling_rate = FS
        tr.stats.network = "NV"
        tr.stats.station = estacion
        tr.stats.channel = "HH" + c
        tr.stats.starttime = t0
        trs.append(tr)
    return Stream(trs)


def detector_stalta(a, canales, cfg=STALTA):
    """STA/LTA clasico. Intervalos nativos; el onset es su inicio."""
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from obspy.signal.filter import bandpass

    n = a.shape[-1]
    salida = {}
    for ch in canales:
        print(f"    STA/LTA canal {ch} ...", flush=True)
        trigs = []
        for ini, fin in segmentos(n):
            x = np.asarray(a[ch, ini:fin], dtype=np.float64)
            if np.allclose(x, 0):
                continue
            x = bandpass(x, cfg["banda"][0], cfg["banda"][1], df=FS,
                         corners=4, zerophase=True)
            cft = classic_sta_lta(x, int(cfg["sta_s"] * FS),
                                  int(cfg["lta_s"] * FS))
            cft = np.nan_to_num(cft, nan=0.0, posinf=0.0, neginf=0.0)
            for t_on, t_off in trigger_onset(cft, cfg["thr_on"],
                                             cfg["thr_off"]):
                trigs.append((ini + int(t_on), ini + int(t_off)))
        ivs = fusionar_canal(trigs, int(FUSION_S * FS))
        salida[ch] = {"intervalos": ivs, "onsets": [s for s, _ in ivs]}
    return salida


def detector_seisbench(a, canales, nombre, modo):
    """
    PhaseNet o EQTransformer (CPU).

    Onsets   : picks de classify(). Se admiten P y S como marcadores de
               momento, porque los eventos volcanicos LP y TR a menudo no
               tienen un arribo P impulsivo identificable.
    Intervalos: solo si el modelo entrega un canal Detection. PhaseNet no
               lo tiene, y por eso queda fuera del Nivel 2.
    """
    import seisbench.models as sbm

    clase = {"phasenet": sbm.PhaseNet, "eqtransformer": sbm.EQTransformer}
    modelo = clase[nombre].from_pretrained(PESOS[nombre])
    modelo.eval()

    n = a.shape[-1]
    salida = {}
    for ch in canales:
        print(f"    {nombre} canal {ch} (componentes={modo}) ...", flush=True)
        trigs, onsets = [], []
        for ini, fin in segmentos(n):
            x = np.asarray(a[ch, ini:fin], dtype=np.float64)
            if np.allclose(x, 0):
                continue
            st = _stream(x, ini, modo, estacion=f"S{ch}")

            # --- onsets: picks ---
            try:
                out = modelo.classify(st)
                for p in getattr(out, "picks", []):
                    conf = float(getattr(p, "peak_value", 1.0) or 0.0)
                    if conf < UMBRAL_PICK:
                        continue
                    t = getattr(p, "peak_time", None)
                    if t is not None:
                        onsets.append(int(round(t.timestamp * FS)))
            except Exception as ex:
                print(f"      aviso: classify() fallo ({type(ex).__name__})")

            # --- intervalos: canal Detection, si existe ---
            ann = modelo.annotate(st)
            canal_det = next((tr for tr in ann
                              if tr.stats.channel.endswith("Detection")), None)
            if canal_det is None:
                continue
            off = int(round(canal_det.stats.starttime.timestamp * FS)) - ini
            curva = np.zeros(fin - ini, dtype=np.float32)
            lo, hi = max(0, off), min(fin - ini, off + canal_det.stats.npts)
            if hi > lo:
                curva[lo:hi] = canal_det.data[lo - off: hi - off]
            for iv in binaria_a_intervalos(curva >= UMBRAL_PROB):
                trigs.append((ini + iv.ini, ini + iv.fin))

        salida[ch] = {"intervalos": fusionar_canal(trigs, int(FUSION_S * FS)),
                      "onsets": sorted(onsets)}
    return salida


# =====================================================================
# INFORME
# =====================================================================

def imprimir_nivel1(nombre, res):
    print(f"\n{'=' * 70}")
    print(f"  {nombre.upper()}  -  NIVEL 1: DETECCION DEL MOMENTO")
    print(f"{'=' * 70}")
    print(f"  Catalogo: {res['n_catalogo']}   Onsets detectados: "
          f"{res['n_onsets']}")
    print(f"\n  {'tol':<7}{'recall':>9}{'precision':>11}{'falsos/h':>11}"
          f"{'err.med':>10}{'err.MAE':>10}")
    for tol in TOL_ONSET_S:
        d = res.get(f"tol_{tol:g}s")
        if not d:
            continue
        print(f"  +-{tol:<4.0f}{d['recall']:>9.3f}{d['precision']:>11.3f}"
              f"{d['onsets_falsos_por_hora']:>11.2f}"
              f"{d['error_mediano_s']:>10.2f}{d['error_mae_s']:>10.2f}")

    ref = res.get(f"tol_{TOL_ONSET_S[-1]:g}s", {}).get("por_clase", {})
    if ref:
        print(f"\n  Recall por clase (tol +-{TOL_ONSET_S[-1]:g} s):")
        print("  " + "  ".join(f"{c}:{d['recall']:.3f}"
                               for c, d in ref.items()))


def imprimir_nivel2(nombre, res):
    print(f"\n{'-' * 70}")
    print(f"  {nombre.upper()}  -  NIVEL 2: CALIDAD DEL RECORTE")
    print(f"{'-' * 70}")
    if res["n_detecciones"] == 0:
        print("  El detector no delimita eventos (sin canal Detection).")
        print("  Queda fuera del Nivel 2 por construccion.")
        return
    print(f"  TP {res['verdaderos_positivos']} | FP {res['falsos_positivos']}"
          f" | Perdidos {res['no_detectados']}")
    print(f"  Recall {res['recall']:.3f} | Precision {res['precision']:.3f}"
          f" | F1 {res['f1']:.3f} | FP/h {res['fp_por_hora']:.2f}")
    print(f"  Onset  : mediana {res['onset_mediana_s']:+.2f} s | "
          f"MAE {res['onset_mae_s']:.2f} s")
    print(f"  Offset : mediana {res['offset_mediana_s']:+.2f} s | "
          f"MAE {res['offset_mae_s']:.2f} s")
    print(f"  Duracion (sesgo mediano): {res['duracion_sesgo_s']:+.2f} s")
    print(f"  IoU    : medio {res['iou_medio']:.3f} | "
          f"mediano {res['iou_mediano']:.3f}")
    print(f"\n  {'clase':<6}{'n':>5}{'recall':>9}{'onsetMAE':>10}"
          f"{'IoU':>8}{'dur.med':>10}")
    for c, d in res["por_clase"].items():
        print(f"  {c:<6}{d['n']:>5}{d['recall']:>9.3f}"
              f"{d['onset_mae_s']:>10.2f}{d['iou_medio']:>8.3f}"
              f"{d['dur_mediana_s']:>10.1f}")


# =====================================================================
# AUTOTEST
# =====================================================================

def autotest():
    print("AUTOTEST\n")
    assert iou(Intervalo(0, 100), Intervalo(0, 100)) == 1.0
    assert iou(Intervalo(0, 100), Intervalo(200, 300)) == 0.0
    print("  [ok] iou")

    ivs = binaria_a_intervalos(np.array([0, 1, 1, 1, 0, 0, 1, 1, 0]))
    assert [(i.ini, i.fin) for i in ivs] == [(1, 4), (6, 8)]
    print("  [ok] binaria_a_intervalos")

    assert fusionar_canal([(0, 10), (12, 20), (100, 110)], 5) == \
        [(0, 20), (100, 110)]
    print("  [ok] fusionar_canal")

    valida = np.ones(200_000, dtype=bool)
    pc = {1: {"intervalos": [(1000, 1300)]},
          2: {"intervalos": [(1050, 1400)]},
          3: {"intervalos": [(90000, 90200)]}}
    d = coincidencia(pc, valida, 2, 5.0, 1.0)
    assert len(d) == 1 and d[0].ini == 1000 and d[0].fin == 1400
    # disparos desfasados SIN solape: la tolerancia es lo que los rescata
    pc2 = {1: {"intervalos": [(1000, 1200)]},
           2: {"intervalos": [(1300, 1500)]}}
    assert len(coincidencia(pc2, valida, 2, 5.0, 1.0)) == 1
    assert len(coincidencia(pc2, valida, 2, 0.5, 1.0)) == 0
    print("  [ok] coincidencia por intervalo con tolerancia")

    pco = {1: {"onsets": [1000, 50000]}, 2: {"onsets": [1120]},
           3: {"onsets": [1050]}}
    o = coincidencia_onsets(pco, valida, 2, 3.0)
    assert o == [1050], o           # mediana del grupo; 50000 va solo
    print("  [ok] coincidencia_onsets usa la mediana y filtra solitarios")

    cat = [Intervalo(1000, 2000, "VT"), Intervalo(5000, 8000, "LP"),
           Intervalo(20000, 21000, "IC")]
    # 1050 esta a 0,5 s de 1000 -> entra en ambas tolerancias
    # 5150 esta a 1,5 s de 5000 -> entra en +-2 s pero no en +-1 s
    r1 = metricas_onset(cat, [1050, 5150, 90000], horas=10.0)
    assert abs(r1["tol_2s"]["recall"] - 2 / 3) < 1e-9, r1["tol_2s"]
    assert abs(r1["tol_1s"]["recall"] - 1 / 3) < 1e-9, r1["tol_1s"]
    print("  [ok] metricas_onset discrimina por tolerancia")

    det = [Intervalo(1050, 2100), Intervalo(5200, 7900),
           Intervalo(50000, 50500)]
    r2, *_ = metricas(cat, det, 10.0)
    assert r2["verdaderos_positivos"] == 2 and r2["falsos_positivos"] == 1
    print("  [ok] metricas de recorte")

    imprimir_nivel1("autotest", r1)
    imprimir_nivel2("autotest", r2)
    print("\nTodos los tests pasaron.")


# =====================================================================
# MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detectores", nargs="+", default=["stalta"],
                   choices=["stalta", "phasenet", "eqtransformer"])
    p.add_argument("--componentes", default=None,
                   choices=["ceros", "duplicar"],
                   help="fuerza el modo; por defecto usa el optimo por modelo")
    p.add_argument("--canales", nargs="+", type=int, default=CANALES)
    p.add_argument("--min-estaciones", type=int, default=MIN_ESTACIONES)
    p.add_argument("--iou-min", type=float, default=IOU_MIN)
    p.add_argument("--autotest", action="store_true")
    args = p.parse_args()

    if args.autotest:
        autotest()
        return

    print("Cargando datos ...")
    a, catalogo, valida, horas = cargar()
    print(f"  Traza   : {a.shape}  ({a.shape[-1] / FS / 3600:.2f} h)")
    print(f"  Validas : {horas:.2f} h | Catalogo: {len(catalogo)} eventos")
    print(f"  Canales : {args.canales} | min_estaciones: "
          f"{args.min_estaciones} | IoU_min: {args.iou_min}")

    todo = {}
    for nombre in args.detectores:
        modo = args.componentes or COMPONENTES_POR_MODELO.get(nombre, "ceros")
        print(f"\n>>> {nombre} (componentes={modo}) ...")

        if nombre == "stalta":
            por_canal = detector_stalta(a, args.canales)
        else:
            por_canal = detector_seisbench(a, args.canales, nombre, modo)

        onsets = coincidencia_onsets(por_canal, valida, args.min_estaciones)
        r1 = metricas_onset(catalogo, onsets, horas)
        imprimir_nivel1(nombre, r1)

        det = coincidencia(por_canal, valida, args.min_estaciones)
        r2, *_ = metricas(catalogo, det, horas, args.iou_min)
        imprimir_nivel2(nombre, r2)

        todo[nombre] = {"componentes": modo, "nivel1": r1, "nivel2": r2}

    os.makedirs(SALIDA, exist_ok=True)
    destino = f"{SALIDA}/metricas_{'_'.join(args.detectores)}.json"
    with open(destino, "w") as f:
        json.dump(todo, f, indent=2, ensure_ascii=False)
    print(f"\nResultados en {destino}")


if __name__ == "__main__":
    main()
