#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ventanas_traza_real.py
======================

Aplica el barrido de 5 minutos (seccion 2.4, tarea B-08) a la traza continua
real con STA/LTA y lo compara con la corrida sobre la traza completa.

Que mide:
    - Nivel 2 (calidad del recorte) global, por clase y por tramo, para la
      traza completa y para el barrido. La hipotesis de la seccion 2.4 es
      que el barrido no cambia la deteccion: STA/LTA calcula su razon
      muestra a muestra y la ventana solo es la unidad de entrega.
    - Cuantas detecciones duran mas de 81,92 s, es decir, cuantas quedarian
      truncadas si el clasificador recibiera la ventana fija del pool.

Que grafica (en resultados/ventanas/):
    - Para un TR largo y un VT corto: cada ventana de 300 s que toca el
      evento, con la onda, el intervalo del catalogo y los disparos de esa
      ventana, mas un panel final con la deteccion fusionada, el recorte
      adaptativo y la ventana fija de 81,92 s.
    - Los espectrogramas del recorte adaptativo y de la ventana fija.

Uso:
    python ventanas_traza_real.py
    python ventanas_traza_real.py --evento 57        # indice en el catalogo

Reutiliza la carga, la configuracion STA/LTA, la coincidencia entre
estaciones y las metricas de evaluacion/evaluar_detectores.py.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
import barrido as br                 # noqa: E402
import evaluar_detectores as ed      # noqa: E402

FS = ed.FS
COLOR = {"VT": "#df8d5e", "LP": "#2ca02c", "TR": "#d62728",
         "AV": "#9467bd", "IC": "#8c564b"}

# Espectrograma ilustrativo; los parametros definitivos se fijan en OE2.
NPERSEG, NOVERLAP = 256, 128


# =====================================================================
# DETECCION
# =====================================================================

def stalta_barrido(a, canales, ventanas):
    """STA/LTA por ventana de 300 s, trasladado a indices globales."""
    det = lambda x: br.stalta_en_ventana(x, ed.STALTA)  # noqa: E731
    por_canal, crudos = {}, {}
    for ch in canales:
        print(f"    barrido STA/LTA canal {ch} ({len(ventanas)} ventanas) ...",
              flush=True)
        c = br.detectar_por_ventanas(a[ch], ventanas, det)
        ivs = br.fusionar([(d["ini"], d["fin"]) for d in c],
                          int(ed.FUSION_S * FS))
        por_canal[ch] = {"intervalos": ivs, "onsets": [s for s, _ in ivs]}
        crudos[ch] = c
    return por_canal, crudos


# =====================================================================
# METRICAS POR AMBITO
# =====================================================================

def tramos(n):
    return ed.segmentos(n)


def num(v, d=1):
    """Numero con coma decimal para los textos de las figuras."""
    return f"{v:.{d}f}".replace(".", ",")


def utc(t):
    return datetime.fromtimestamp(float(t), timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S")


def resumir(catalogo, det, horas):
    res, _, _, _ = ed.metricas(catalogo, det, horas)
    largas = sum(1 for d in det if d.dur / FS > br.VENTANA_FIJA_S)
    return {"n_catalogo": res["n_catalogo"], "n_detecciones": res["n_detecciones"],
            "recall": res["recall"], "precision": res["precision"],
            "fp_por_hora": res["fp_por_hora"], "iou_medio": res["iou_medio"],
            "detecciones_mas_de_81_92_s": largas,
            "por_clase": res["por_clase"]}


def por_tramo(a, catalogo, det, valida):
    n = a.shape[-1]
    salida = []
    for k, (t0, t1) in enumerate(tramos(n), start=1):
        cat = [e for e in catalogo if t0 <= e.ini < t1]
        dd = [d for d in det if t0 <= d.ini < t1]
        horas = float(valida[t0:t1].sum()) / FS / 3600.0
        r = resumir(cat, dd, horas)
        r.update({"tramo": k, "muestra_inicial": int(t0), "muestras": int(t1 - t0),
                  "inicio_utc": utc(a[0, t0]), "termino_utc": utc(a[0, t1 - 1])})
        salida.append(r)
    return salida


# =====================================================================
# FIGURAS
# =====================================================================

def elegir_canal(a, ev, canales):
    """Canal con mayor razon de energia entre el evento y los 60 s previos."""
    mejor, r_max = canales[0], -1.0
    pre = int(60 * FS)
    for ch in canales:
        x = np.asarray(a[ch, max(0, ev.ini - pre):ev.fin], dtype=np.float64)
        e = np.mean(x[-(ev.fin - ev.ini):] ** 2)
        r = np.mean(x[:max(1, len(x) - (ev.fin - ev.ini))] ** 2)
        if r > 0 and e / r > r_max:
            mejor, r_max = ch, e / r
    return mejor


def elegir_eventos(catalogo, det):
    """TR mas largo y VT mas corto que tengan alguna deteccion solapada."""
    def detectado(e):
        return any(d.ini < e.fin and e.ini < d.fin for d in det)
    tr = sorted([e for e in catalogo if e.clase == "TR" and detectado(e)],
                key=lambda e: -e.dur)
    vt = sorted([e for e in catalogo if e.clase == "VT" and detectado(e)],
                key=lambda e: e.dur)
    return [x[0] for x in (tr, vt) if x]


def figura_ventanas(a, ev, ch, ventanas, crudos, det, destino):
    import figuras

    n = a.shape[-1]
    rel = lambda m: (m - ev.ini) / FS  # noqa: E731
    k_toca = br.ventanas_que_tocan(ev.ini, ev.fin, ventanas)
    lo, hi = ventanas[k_toca[0]][0], ventanas[k_toca[-1]][1]
    disparos = {}
    for c in crudos[ch]:
        if c["ventana"] in k_toca:
            disparos.setdefault(c["ventana"], []).append(
                (rel(c["ini"]), rel(c["fin"]), c["toca_borde_izq"] or c["toca_borde_der"]))
    solap = [d for d in det if d.ini < ev.fin and ev.ini < d.fin]
    ra = rf = None
    if solap:
        d = max(solap, key=lambda d: ed.iou(ev, d))
        r0, r1 = br.recorte_adaptativo(d.ini, d.fin, n, ed.CORTES)
        f0, f1 = br.recorte_fijo(d.ini, n, ed.CORTES)
        ra, rf = (rel(r0), rel(r1)), (rel(f0), rel(f1))
    figuras.figura_barrido(
        t=(np.arange(lo, hi) - ev.ini) / FS,
        x=np.asarray(a[ch, lo:hi], dtype=np.float64),
        ref=(0.0, ev.dur / FS), color_ref=COLOR.get(ev.clase, "0.5"),
        etiqueta_ref=f"catálogo: {ev.clase} de {num(ev.dur / FS)} s",
        ventanas=[(k, rel(ventanas[k][0]), rel(ventanas[k][1])) for k in k_toca],
        disparos=disparos,
        contenedoras=set(br.ventanas_que_contienen(ev.ini, ev.fin, ventanas)),
        fusionados=[(rel(d.ini), rel(d.fin)) for d in solap],
        recorte_adapt=ra, recorte_fijo=rf,
        titulo=f"{ev.clase} de {num(ev.dur / FS)} s, canal {ch}, inicio "
               f"{utc(a[0, ev.ini])} UTC. STA/LTA con ventanas de 300 s y paso de"
               " 55 s; disparos del canal y detección de red",
        destino=destino)
    return solap


def figura_espectrogramas(a, ev, ch, det, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    n = a.shape[-1]
    solap = [d for d in det if d.ini < ev.fin and ev.ini < d.fin]
    if not solap:
        return False
    d = max(solap, key=lambda d: ed.iou(ev, d))
    recortes = [("recorte adaptativo", br.recorte_adaptativo(d.ini, d.fin, n, ed.CORTES)),
                ("ventana fija de 81,92 s", br.recorte_fijo(d.ini, n, ed.CORTES))]
    fig, ejes = plt.subplots(2, 2, figsize=(10, 4.6),
                             gridspec_kw={"height_ratios": [1, 2]})
    for j, (nombre, (r0, r1)) in enumerate(recortes):
        x = np.asarray(a[ch, r0:r1], dtype=np.float64)
        tt = np.arange(r0, r1) / FS - r0 / FS
        ejes[0, j].plot(tt, x, lw=0.3, color="0.25")
        c0 = max(0, (ev.ini - r0) / FS)
        c1 = min((r1 - r0) / FS, (ev.fin - r0) / FS)
        ejes[0, j].axvspan(c0, c1, color=COLOR.get(ev.clase, "0.5"), alpha=0.15)
        ejes[0, j].set_title(f"{nombre}: {num((r1 - r0) / FS)} s", fontsize=8)
        ejes[0, j].set_yticks([])
        f, ts, S = spectrogram(x, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        m = f <= 20
        ejes[1, j].pcolormesh(ts, f[m], 10 * np.log10(S[m] + 1e-20),
                              shading="auto", cmap="viridis")
        ejes[1, j].set_xlabel("tiempo desde el inicio del recorte (s)")
        ejes[1, j].set_ylabel("Hz")
    fuera = [max(0.0, (ev.fin - r1) / FS) + max(0.0, (r0 - ev.ini) / FS)
             for _, (r0, r1) in recortes]
    fig.suptitle(f"{ev.clase} de {num(ev.dur / FS)} s, canal {ch}. Quedan fuera del "
                 f"recorte {num(fuera[0])} s del evento del catálogo con el recorte "
                 f"adaptativo y {num(fuera[1])} s con la ventana fija", fontsize=8)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return True


# =====================================================================
# INFORME
# =====================================================================

def fila(nombre, r):
    return (f"  {nombre:<22}{r['n_catalogo']:>5}{r['n_detecciones']:>6}"
            f"{r['recall']:>8.3f}{r['precision']:>8.3f}{r['fp_por_hora']:>8.2f}"
            f"{r['iou_medio']:>8.3f}{r['detecciones_mas_de_81_92_s']:>8}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default=ed.NPY)
    ap.add_argument("--csv", default=ed.CSV)
    ap.add_argument("--salida", default=os.path.join(ed.SALIDA, "ventanas"))
    ap.add_argument("--evento", type=int, default=None,
                    help="indice en el catalogo filtrado; por defecto TR y VT")
    args = ap.parse_args(argv)
    ed.NPY, ed.CSV = args.npy, args.csv
    os.makedirs(args.salida, exist_ok=True)

    print("Cargando traza y catalogo ...")
    a, catalogo, valida, horas = ed.cargar()
    n = a.shape[-1]
    ventanas = br.generar_ventanas(n, ed.CORTES)
    print(f"  {len(catalogo)} eventos, {horas:.2f} h validas, {len(ventanas)} ventanas")

    print("STA/LTA sobre la traza completa ...")
    pc_c = ed.detector_stalta(a, ed.CANALES)
    det_c = ed.coincidencia(pc_c, valida)
    print("STA/LTA con barrido de 5 minutos ...")
    pc_b, crudos = stalta_barrido(a, ed.CANALES, ventanas)
    det_b = ed.coincidencia(pc_b, valida)

    res = {"config": {"ventana_s": br.VENTANA_S, "paso_s": br.PASO_S,
                      "gap_fusion_s": br.GAP_FUSION_S,
                      "margen_pre_s": br.MARGEN_PRE_S, "stalta": ed.STALTA,
                      "canales": ed.CANALES, "n_ventanas": len(ventanas)}}
    for nombre, det in (("completa", det_c), ("barrido", det_b)):
        res[nombre] = {"global": resumir(catalogo, det, horas),
                       "por_tramo": por_tramo(a, catalogo, det, valida)}
    iguales = sorted((d.ini, d.fin) for d in det_c) == sorted((d.ini, d.fin) for d in det_b)
    res["detecciones_identicas"] = iguales
    dur_cat = {c: [e.dur / FS for e in catalogo if e.clase == c] for c in ed.CLASES}
    res["catalogo_mas_de_81_92_s"] = {c: sum(1 for d in v if d > br.VENTANA_FIJA_S)
                                      for c, v in dur_cat.items()}

    ruta = os.path.join(args.salida, "comparacion_barrido.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 78)
    print("  NIVEL 2 (IoU >= 0,30)   cat   det  recall   prec    FP/h  IoU med  >81,92s")
    print("=" * 78)
    for modo in ("completa", "barrido"):
        print(fila(f"{modo}: global", res[modo]["global"]))
        for r in res[modo]["por_tramo"]:
            print(fila(f"{modo}: tramo {r['tramo']}", r))
    print(f"\n  Detecciones identicas entre completa y barrido: {iguales}")
    print("\n  Recall e IoU por clase (completa | barrido):")
    for c in ed.CLASES:
        pc, pb = (res[m]["global"]["por_clase"].get(c) for m in ("completa", "barrido"))
        if pc:
            print(f"    {c}: recall {pc['recall']:.3f} | {pb['recall']:.3f}   "
                  f"IoU {pc['iou_medio']:.3f} | {pb['iou_medio']:.3f}   "
                  f"catalogo >81,92 s: {res['catalogo_mas_de_81_92_s'][c]} de {pc['n']}")
    print("\n  Tramos (hora leida de la fila 0 de la traza):")
    for r in res["completa"]["por_tramo"]:
        print(f"    tramo {r['tramo']}: {r['inicio_utc']} -> {r['termino_utc']} UTC,"
              f" {r['n_catalogo']} eventos")

    if args.evento is not None:
        eventos = [catalogo[args.evento]]
    else:
        eventos = elegir_eventos(catalogo, det_b)
    for ev in eventos:
        ch = elegir_canal(a, ev, ed.CANALES)
        base = os.path.join(args.salida, f"{ev.clase}_{ev.ini}")
        figura_ventanas(a, ev, ch, ventanas, crudos, det_b, base + "_ventanas.png")
        figura_espectrogramas(a, ev, ch, det_b, base + "_espectrogramas.png")
        print(f"\n  Figuras de {ev.clase} ({ev.dur / FS:.1f} s, canal {ch}): {base}_*.png")
    print(f"  Resultados: {ruta}")


if __name__ == "__main__":
    main()
