#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figuras de la evaluacion con FRE.

Uso: python figuras_fre3c.py [--umbral 0.10]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fre3c_lib as lib

FS = lib.FS
TOL = int(round(lib.TOL_S * FS))
CANAL_FRE = 1  # fila de FRE en la traza de Zenodo
DIR_CACHE = os.path.join(lib.DIR_TESIS, "resultados")

COLOR = {"PN-PS": "#6250d6", "PN-P": "#e87ba4", "PN-S": "#008300", "EQT-D": "#eda100"}
NOMBRE = {"PN-PS": "PhaseNet P y S", "PN-P": "PhaseNet solo P", "PN-S": "PhaseNet solo S",
          "EQT-D": "EQTransformer (deteccion)"}
TINTA, TINTA2, TINTA3, REJILLA, FONDO = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def coma(v, _pos=None):
    # decimales con coma, como en el informe
    return f"{v:g}".replace(".", ",")


def estilo(ax):
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(coma))
    ax.yaxis.set_major_formatter(FuncFormatter(coma))
    ax.grid(axis="y", color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TINTA2)
    ax.tick_params(colors=TINTA2, labelsize=9)


def guardar(fig, nombre):
    ruta = os.path.join(lib.DIR_SALIDA, nombre)
    fig.savefig(ruta, dpi=160, facecolor=FONDO)
    print(f"  {ruta}")


def fila(sc, flujo, filtro, umbral, clase, tramo="todos"):
    r = sc[(sc.flujo == flujo) & (sc.filtro == filtro) & np.isclose(sc.umbral, umbral)
           & (sc.tramo.astype(str) == tramo) & (sc.clase == clase)]
    return r.iloc[0] if len(r) else None


def fig1(plt, sc, umbral, filtro="f1_15"):
    grupos = ["VT", "LP", "TR", "volcanico", "AV", "IC"]
    etiquetas = ["VT", "LP", "TR", "Total\nvolcanico", "AV", "IC"]
    flujos = ["PN-PS", "EQT-D"]
    x = np.array([0, 1, 2, 3.2, 4.6, 5.6])
    ancho = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, fl in enumerate(flujos):
        for xi, g in zip(x, grupos):
            r = fila(sc, fl, filtro, umbral, g)
            if r is None:
                continue
            pos = xi + (i - 0.5) * ancho
            ax.bar(pos, r.sens, ancho * 0.92, color=COLOR[fl], label=NOMBRE[fl] if g == "VT" else None)
            ax.errorbar(pos, r.sens, yerr=[[max(0.0, r.sens - r.ic_inf)], [max(0.0, r.ic_sup - r.sens)]],
                        fmt="none", ecolor=TINTA2, elinewidth=1, capsize=2)
            ax.text(pos, r.ic_sup + 0.02, f"{int(r.k)}/{int(r.n)}", ha="center", va="bottom",
                    fontsize=7.5, color=TINTA2)
    ax.axvline(4.0, color=TINTA3, lw=0.8, ls=":")
    ax.text(5.1, 1.13, "no volcanicos (van al rechazo)", ha="center", fontsize=8.5, color=TINTA2)
    ax.text(1.6, 1.13, "volcanicos (metrica principal)", ha="center", fontsize=8.5, color=TINTA2)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("sensibilidad (aciertos / eventos)", color=TINTA2, fontsize=9)
    estilo(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, color=TINTA)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title(f"Que fraccion de cada clase detecta cada red (FRE, 3 componentes, "
                 f"umbral {umbral:.2f}, filtro 1-15 Hz)".replace(".", ","), fontsize=10.5, color=TINTA, loc="left")
    fig.text(0.01, 0.01, "Barras finas: intervalo de confianza del 95 %. Sobre cada barra: "
             "eventos detectados / eventos de la clase.", fontsize=7.5, color=TINTA3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    guardar(fig, "fig1_por_clase.png")
    plt.close(fig)


def fig2(plt, sc, st, filtro="f1_15"):
    fig, ax = plt.subplots(figsize=(9, 5))
    marcar = [0.05, 0.10, 0.30, 0.50]
    for fl in ["PN-PS", "PN-P", "PN-S", "EQT-D"]:
        v = sc[(sc.flujo == fl) & (sc.filtro == filtro) & (sc.tramo.astype(str) == "todos")
               & (sc.clase == "volcanico")][["umbral", "sens"]]
        t = st[(st.flujo == fl) & (st.filtro == filtro) & (st.tramo.astype(str) == "todos")][["umbral", "fp_h"]]
        d = v.merge(t, on="umbral").sort_values("umbral")
        if d.empty:
            continue
        ax.plot(d.fp_h, d.sens, color=COLOR[fl], lw=2, marker="o", ms=3.5, label=NOMBRE[fl])
        if fl == "PN-PS":
            for u in marcar:
                r = d[np.isclose(d.umbral, u)]
                if len(r):
                    ax.annotate(f"umbral {u:.2f}".replace(".", ","), (r.fp_h.iloc[0], r.sens.iloc[0]),
                                xytext=(6, -12), textcoords="offset points", fontsize=7.5,
                                color=TINTA2)
    ax.axhline(0.95, color=TINTA2, lw=0.8, ls=":")
    ax.text(0.01, 0.95, "meta 0,95", ha="left", va="bottom", fontsize=8, color=TINTA2,
            transform=ax.get_yaxis_transform())
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("falsos positivos por hora", color=TINTA2, fontsize=9)
    ax.set_ylabel("sensibilidad volcanica (VT + LP + TR)", color=TINTA2, fontsize=9)
    estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Cuantas falsas alarmas cuesta cada nivel de sensibilidad (FRE, filtro 1-15 Hz)\n"
                 "Cada punto es un umbral; mas arriba y mas a la izquierda es mejor",
                 fontsize=10.5, color=TINTA, loc="left")
    fig.tight_layout()
    guardar(fig, "fig2_costo_fp.png")
    plt.close(fig)


def sens_cache_picks(modelo, modo, inc, zona_ini, zona_fin, valida, umbrales):
    ruta = os.path.join(DIR_CACHE, "cache_picks", f"picks_{modelo}_{modo}.csv")
    if not os.path.exists(ruta):
        print(f"  (falta {ruta}: se omite esa curva)")
        return None
    p = pd.read_csv(ruta)
    p = p[p.canal == CANAL_FRE]
    filas = []
    for u in umbrales:
        o = lib.agrupar_anclado(p[p.conf >= u].muestra.to_numpy(), TOL)
        res = lib.evaluar(o, inc, zona_ini, zona_fin, valida)
        f, _ = lib.resumir(res, inc, 1.0)
        filas.append(dict(umbral=u, **{r["clase"]: r["sens"] for r in f}))
    return pd.DataFrame(filas)


def sens_cache_deteccion(modo, inc, zona_ini, zona_fin, valida, umbrales):
    ruta = os.path.join(DIR_CACHE, "cache_detection", f"detection_eqt_{modo}.npz")
    if not os.path.exists(ruta):
        print(f"  (falta {ruta}: se omite esa curva)")
        return None
    z = np.load(ruta)
    if str(CANAL_FRE) not in z.files:
        print(f"  (la cache {ruta} no trae el canal {CANAL_FRE})")
        return None
    curva = z[str(CANAL_FRE)]
    filas = []
    for u in umbrales:
        ini, _, _, _ = lib.disparos_por_tramo(curva, u)
        res = lib.evaluar(ini, inc, zona_ini, zona_fin, valida)
        f, _ = lib.resumir(res, inc, 1.0)
        filas.append(dict(umbral=u, **{r["clase"]: r["sens"] for r in f}))
    return pd.DataFrame(filas)


def fig3(plt, sc, eventos, valida):
    inc = eventos[eventos.incluido].reset_index(drop=True)
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zi, zf = (con_senal.ini - TOL).to_numpy(), con_senal.fin.to_numpy()
    umbrales = [u for u in lib.UMBRALES if u >= 0.05]

    paneles = [
        ("PhaseNet, arribos P y S", "PN-PS-ant",
         lambda m: sens_cache_picks("phasenet", m, inc, zi, zf, valida, umbrales)),
        ("EQTransformer, curva de deteccion", "EQT-D",
         lambda m: sens_cache_deteccion(m, inc, zi, zf, valida, umbrales)),
    ]
    estilos = {"ceros": dict(color=TINTA3, ls="--", lw=2, label="Z + horizontales en cero (Etapa 2)"),
               "duplicar": dict(color=TINTA2, ls=":", lw=2.2, label="Z duplicada en las tres"),
               "reales": dict(lw=2.6, label="Z, N y E reales")}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    tabla = []
    for ax, (titulo, flujo_nuevo, f_cache) in zip(axs, paneles):
        for modo in ("ceros", "duplicar"):
            d = f_cache(modo)
            if d is None:
                continue
            ax.plot(d.umbral, d.volcanico, **estilos[modo])
            for u in (0.05, 0.10, 0.30):
                r = d[np.isclose(d.umbral, u)]
                if len(r):
                    tabla.append(dict(red=titulo, entrada=modo, umbral=u,
                                      **{c: r[c].iloc[0] for c in lib.VOLCANICAS + ["volcanico"]}))
        n = sc[(sc.flujo == flujo_nuevo) & (sc.filtro == "f1_15") & (sc.tramo.astype(str) == "todos")]
        piv = n.pivot_table(index="umbral", columns="clase", values="sens").reset_index()
        piv = piv[piv.umbral >= 0.05]
        color = COLOR["PN-PS"] if flujo_nuevo.startswith("PN") else COLOR["EQT-D"]
        ax.plot(piv.umbral, piv.volcanico, color=color, **estilos["reales"])
        for u in (0.05, 0.10, 0.30):
            r = piv[np.isclose(piv.umbral, u)]
            if len(r):
                tabla.append(dict(red=titulo, entrada="reales", umbral=u,
                                  **{c: r[c].iloc[0] for c in lib.VOLCANICAS + ["volcanico"]}))
        ax.axhline(0.95, color=TINTA2, lw=0.8, ls=":")
        ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")
        ax.set_xlabel("umbral de confianza", color=TINTA2, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        estilo(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axs[0].set_ylabel("sensibilidad volcanica (VT + LP + TR)", color=TINTA2, fontsize=9)
    fig.suptitle("Misma estacion (FRE), mismos eventos, mismo procedimiento: solo cambia lo que "
                 "recibe la red", fontsize=10.5, color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    guardar(fig, "fig3_antes_ahora.png")
    plt.close(fig)
    if tabla:
        t = pd.DataFrame(tabla)
        print("\nANTES Y AHORA EN FRE (sensibilidad por clase, filtro 1-15 Hz):")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        t.to_csv(os.path.join(lib.DIR_SALIDA, "antes_ahora_fre.csv"), index=False)


def fig4(plt, sc):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, fl in zip(axs, ["PN-PS", "EQT-D"]):
        for filtro, ls in (("f1_15", "-"), ("f1_45", "--")):
            d = sc[(sc.flujo == fl) & (sc.filtro == filtro) & (sc.tramo.astype(str) == "todos")
                   & (sc.clase == "volcanico")].sort_values("umbral")
            if d.empty:
                continue
            ax.plot(d.umbral, d.sens, color=COLOR[fl], ls=ls, lw=2.2,
                    label=f"filtro {filtro[1:].replace('_', '-')} Hz")
        ax.axhline(0.95, color=TINTA2, lw=0.8, ls=":")
        ax.set_title(NOMBRE[fl], fontsize=10, color=TINTA, loc="left")
        ax.set_xlabel("umbral de confianza", color=TINTA2, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        estilo(ax)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    axs[0].set_ylabel("sensibilidad volcanica (VT + LP + TR)", color=TINTA2, fontsize=9)
    fig.suptitle("Filtro 1-15 Hz contra 1-45 Hz (la banda con que se entreno EQTransformer)",
                 fontsize=10.5, color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    guardar(fig, "fig4_filtros.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.10, help="umbral de la figura 1")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = pd.read_csv(os.path.join(lib.DIR_SALIDA, "sens_clase.csv"))
    st = pd.read_csv(os.path.join(lib.DIR_SALIDA, "sens_totales.csv"))
    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    valida = np.load(lib.ruta_datos("f1_15"))["valida"]

    print("Figuras:")
    fig1(plt, sc, args.umbral)
    fig2(plt, sc, st)
    fig3(plt, sc, eventos, valida)
    fig4(plt, sc)


if __name__ == "__main__":
    main()
