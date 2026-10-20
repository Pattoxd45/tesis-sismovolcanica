#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calidad del recorte con varios umbrales de IoU y los cuatro casos de cruce entre recorte
y evento, igual para todos los recortes.

Uso: python iou_fre3c.py [--umbral 0.10] [--filtro f1_15] [--figura]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
import fre3c_lib as lib
import evaluar_detectores as ed
import recorte_fre3c as rc

FS = lib.FS
UMBRALES_IOU = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
UMBRALES_CLASE = [0.3, 0.5, 0.7]
CASOS = ["cubre todo", "le falta el final", "le faltan inicio y fin", "le falta el inicio"]
RECORTES = ["PhaseNet + energia con corte", "PhaseNet + energia", "PhaseNet + ventana fija",
            "EQTransformer, propio", "STA/LTA, propio"]
TOCA = 1e-9  # cualquier tiempo en comun cuenta como cruce


def recortes_todos(umbral, filtro, valida):
    z = np.load(lib.ruta_datos(filtro))
    env = rc.envolvente(z["Z"], z["N"], z["E"])
    pn = np.load(lib.ruta_anot("phasenet", filtro))
    o = rc.onsets_pn_ps({k: pn[k] for k in ("P", "S")}, umbral, valida)
    det = np.load(lib.ruta_anot("eqtransformer", filtro))["Detection"]
    return {
        "PhaseNet + energia con corte": (rc.recortes(o, env, valida, rc.K_RUIDO, corte=True), rc.MARGEN_PREVIO_S),
        "PhaseNet + energia": (rc.recortes(o, env, valida, rc.K_RUIDO), rc.MARGEN_PREVIO_S),
        "PhaseNet + ventana fija": (rc.recortes(o, env, valida, fija=True), rc.MARGEN_PREVIO_S),
        "EQTransformer, propio": (rc.recortes_eqt(det, umbral, valida), 0.0),
        "STA/LTA, propio": (rc.recortes_stalta(rc.cft_stalta(z["Z"]), valida, ed.STALTA["thr_on"]), 0.0),
    }


def caso(g, d):
    if d.ini <= g.ini and d.fin >= g.fin:
        return 0
    if d.ini <= g.ini:
        return 1
    if d.fin < g.fin:
        return 2
    return 3


def por_evento(nombre, ivs, inc):
    cat = [ed.Intervalo(int(r.ini), int(r.fin), r.clase) for r in inc.itertuples()]
    det = [ed.Intervalo(a, b) for a, b in ivs]
    # emparejar ordena por IoU de mayor a menor: contar IoU >= u aqui da lo mismo que emparejar con u
    pares, _, _ = ed.emparejar(cat, det, TOCA)
    par = {id(g): (d, v) for g, d, v in pares}
    filas = []
    for g, r in zip(cat, inc.itertuples()):
        d, v = par.get(id(g), (None, 0.0))
        filas.append(dict(recorte=nombre, id=int(r.id), clase=g.clase, dur_evento_s=g.dur / FS,
                          iou=v, caso=CASOS[caso(g, d)] if d is not None else "sin cruce",
                          dur_recorte_s=d.dur / FS if d is not None else np.nan))
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.10)
    ap.add_argument("--filtro", default="f1_15", choices=list(lib.FILTROS))
    ap.add_argument("--figura", action="store_true")
    args = ap.parse_args()

    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    valida = np.load(lib.ruta_datos("f1_15"))["valida"]
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    tol = int(round(lib.TOL_S * FS))
    zi, zf = (con_senal.ini - tol).to_numpy(), con_senal.fin.to_numpy()
    horas = valida.sum() / FS / 3600
    todos = recortes_todos(args.umbral, args.filtro, valida)

    pe = pd.DataFrame([f for n in RECORTES for f in por_evento(n, todos[n][0], inc)])
    volc = pe[pe.clase.isin(lib.VOLCANICAS)]
    n_volc = int((inc.clase.isin(lib.VOLCANICAS)).sum())

    print(f"CALIDAD DEL RECORTE (redes con umbral {args.umbral:.2f}, filtro {args.filtro}; "
          "STA/LTA con la configuracion de la Etapa 2)")
    print("  IoU = tiempo en comun / tiempo total entre recorte y evento; un recorte por evento")
    resumen = []
    for n in RECORTES:
        fc, tot = rc.evaluar_nivel2(todos[n][0], inc, zi, zf, horas, todos[n][1])
        c = {f["clase"]: f for f in fc}
        k30 = int((volc[volc.recorte == n].iou >= 0.3).sum())
        print(f"  comprobacion {n}: IoU >= 0,30 igual a recorte_fre3c.py: "
              f"{'SI' if k30 == c['volcanico']['k'] else 'NO'}")
        resumen.append(dict(recorte=n, **{f"k_{k}": c[k]["k"] for k in lib.VOLCANICAS},
                            k_volc=c["volcanico"]["k"], iou_medio=c["volcanico"]["iou_medio"],
                            fp_h=tot["fp_h"], dur_mediana_s=tot["dur_mediana_volc_s"]))

    print(f"\n1. VOLCANICOS CON IoU SOBRE CADA UMBRAL (de {n_volc}); 'toca' = cualquier cruce")
    print(f"   {'recorte':<30} {'toca':>5} " + " ".join(f"{u:>5.1f}" for u in UMBRALES_IOU))
    for n in RECORTES:
        v = volc[volc.recorte == n].iou
        print(f"   {n:<30} {int((v > 0).sum()):>5} " + " ".join(f"{int((v >= u).sum()):>5}" for u in UMBRALES_IOU))

    print("\n2. POR CLASE")
    print(f"   {'recorte':<30} {'clase':>5} {'n':>4} {'toca':>5} " + " ".join(f"{u:>5.1f}" for u in UMBRALES_CLASE)
          + f" {'IoU mediano':>12}")
    for n in RECORTES:
        for cl in lib.VOLCANICAS:
            v = pe[(pe.recorte == n) & (pe.clase == cl)].iou
            med = v[v > 0].median() if (v > 0).any() else np.nan
            print(f"   {n:<30} {cl:>5} {len(v):>4} {int((v > 0).sum()):>5} "
                  + " ".join(f"{int((v >= u).sum()):>5}" for u in UMBRALES_CLASE) + f" {med:>12.2f}")
        print()

    print("3. CUATRO CASOS DE CRUCE (volcanicos con algun cruce): cantidad e IoU mediano")
    print(f"   {'recorte':<30} " + " ".join(f"{c:>24}" for c in CASOS + ["sin cruce"]))
    for n in RECORTES:
        g = volc[volc.recorte == n]
        celdas = []
        for c in CASOS:
            x = g[g.caso == c].iou
            celdas.append(f"{len(x):>3} (IoU {x.median():.2f})" if len(x) else f"{0:>3}")
        celdas.append(f"{int((g.caso == 'sin cruce').sum()):>3}")
        print(f"   {n:<30} " + " ".join(f"{c:>24}" for c in celdas))

    print("\n4. RESUMEN CON IoU >= 0,30 (para el cuadro de recortes)")
    print(f"   {'recorte':<30} {'VT':>4} {'LP':>4} {'TR':>4} {'volc':>5} {'IoU medio':>9} {'FP/h':>6} {'dur med (s)':>11}")
    for r in resumen:
        print(f"   {r['recorte']:<30} {r['k_VT']:>4} {r['k_LP']:>4} {r['k_TR']:>4} {r['k_volc']:>5} "
              f"{r['iou_medio']:>9.2f} {r['fp_h']:>6.1f} {r['dur_mediana_s']:>11.1f}")

    # detection_threshold es el unico parametro de EQTransformer que cambia su recorte (se apaga en la mitad)
    det = np.load(lib.ruta_anot("eqtransformer", args.filtro))["Detection"]
    print("\n5. EQTRANSFORMER: RECORTE PROPIO SEGUN detection_threshold (volcanicos de "
          f"{n_volc})")
    print(f"   {'umbral':>6} {'toca':>5} {'0.3':>5} {'0.5':>5} {'IoU mediano':>12} {'dur med (s)':>11} {'FP/h':>6}")
    filas_u = []
    for u in lib.UMBRALES:
        ivs = rc.recortes_eqt(det, u, valida)
        v = pd.DataFrame(por_evento("eqt", ivs, inc))
        v = v[v.clase.isin(lib.VOLCANICAS)].iou
        _, tot = rc.evaluar_nivel2(ivs, inc, zi, zf, horas, 0.0)
        fila = dict(detection_threshold=u, toca=int((v > 0).sum()), k30=int((v >= 0.3).sum()),
                    k50=int((v >= 0.5).sum()), iou_mediano=v[v > 0].median() if (v > 0).any() else np.nan,
                    dur_mediana_s=tot["dur_mediana_volc_s"], fp_h=tot["fp_h"])
        filas_u.append(fila)
        print(f"   {u:>6.2f} {fila['toca']:>5} {fila['k30']:>5} {fila['k50']:>5} {fila['iou_mediano']:>12.2f} "
              f"{fila['dur_mediana_s']:>11.1f} {fila['fp_h']:>6.1f}")
    pd.DataFrame(filas_u).to_csv(os.path.join(lib.DIR_SALIDA, "iou_eqt_umbral.csv"), index=False)

    pe.to_csv(os.path.join(lib.DIR_SALIDA, "iou_por_evento.csv"), index=False)
    pd.DataFrame(resumen).to_csv(os.path.join(lib.DIR_SALIDA, "iou_resumen.csv"), index=False)
    print(f"\nTablas en {lib.DIR_SALIDA}/iou_por_evento.csv y iou_resumen.csv")
    if args.figura:
        figura(volc, n_volc)


def figura(volc, n_volc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # orden fijo de la paleta categorica validada (azul, naranjo, aqua, amarillo, magenta)
    color = dict(zip(RECORTES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
    tinta, tinta2, rejilla = "#0b0b0b", "#52514e", "#e4e3df"
    nombre = {"PhaseNet + energia con corte": "PhaseNet + energía con corte",
              "PhaseNet + energia": "PhaseNet + energía",
              "PhaseNet + ventana fija": "PhaseNet + ventana fija de 81,92 s",
              "EQTransformer, propio": "EQTransformer, recorte propio",
              "STA/LTA, propio": "STA/LTA, recorte propio"}
    u = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axvline(0.3, color=tinta2, lw=0.8, ls=":", zorder=0)
    ax.text(0.305, 1.0, "0,30", color=tinta2, fontsize=8, va="top")
    for n in RECORTES:
        v = volc[volc.recorte == n].iou.to_numpy()
        y = [(v > 0).sum() / n_volc if x == 0 else (v >= x).sum() / n_volc for x in u]
        ax.plot(u, y, color=color[n], lw=2, label=nombre[n])
    coma = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(coma)
    ax.yaxis.set_major_formatter(coma)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("umbral de IoU", color=tinta2, fontsize=9)
    ax.set_ylabel("parte de los volcánicos con IoU ≥ umbral", color=tinta2, fontsize=9)
    ax.grid(axis="y", color=rejilla, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(tinta2)
    ax.tick_params(colors=tinta2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    ruta = os.path.join(lib.DIR_SALIDA, "fig7_iou.png")
    fig.savefig(ruta, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Figura: {ruta}")


if __name__ == "__main__":
    main()
