#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensibilidad_fre3c.py
=====================

Tabla de sensibilidad por clase (prioridad de la revision) para la
deteccion con FRE y sus tres componentes reales, por detector, filtro y
umbral, con falsos positivos por hora y tasa de falsos descubrimientos.

Es el NIVEL 1, deteccion del inicio: un evento cuenta como detectado si una
deteccion cae a menos de 5 s de su inicio en el catalogo (asignacion uno a
uno). Se mide sobre la traza continua de FRE, contra el catalogo.

Metrica principal: sensibilidad sobre los eventos volcanicos (VT, LP, TR).
AV e IC se reportan aparte, con su propia sensibilidad, porque iran a la
clase de rechazo del clasificador: sus aciertos no son falsos positivos ni
entran en la sensibilidad volcanica. Definiciones en fre3c_lib.py.

Flujos de deteccion:
    PN-P    arribos P de PhaseNet          PN-S    arribos S de PhaseNet
    PN-PS   P y S unidos (agrupacion anclada de 5 s, gana el mas temprano)
    EQT-D   inicio de cada intervalo de la curva de deteccion de EQTransformer
    EQT-P, EQT-S, EQT-PS   igual que PhaseNet, con los arribos de EQTransformer

Auditoria del calculo anterior: PN-PS-ant y EQT-PS-ant extraen los arribos
como lo hacia la cache de barrido_modelos.py (classify con umbral 0,02 y
filtro posterior por confianza). La diferencia con PN-PS y EQT-PS, sobre
los MISMOS datos, mide el sesgo de ese procedimiento.

Salidas en ~/tesis/resultados/fre3c/:
    sens_clase.csv     una fila por flujo x filtro x umbral x tramo x clase
    sens_totales.csv   una fila por flujo x filtro x umbral x tramo
    curvas_sens_fdr.png y curva_<flujo>_<filtro>.png
    resumen_fre3c.txt  lo mismo que se imprime

Uso:
    python sensibilidad_fre3c.py

Autor: Patricio Valdes Troncoso - Universidad Catolica de Temuco
"""

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fre3c_lib as lib  # noqa: E402

FS = lib.FS
TOL = int(round(lib.TOL_S * FS))

# flujo -> (modelo, etiquetas de curva, forma de extraer el instante)
FLUJOS = {
    "PN-P": ("phasenet", ["P"], "pico"),
    "PN-S": ("phasenet", ["S"], "pico"),
    "PN-PS": ("phasenet", ["P", "S"], "pico"),
    "EQT-D": ("eqtransformer", ["Detection"], "inicio"),
    "EQT-P": ("eqtransformer", ["P"], "pico"),
    "EQT-S": ("eqtransformer", ["S"], "pico"),
    "EQT-PS": ("eqtransformer", ["P", "S"], "pico"),
    "PN-PS-ant": ("phasenet", ["P", "S"], "cache"),
    "EQT-PS-ant": ("eqtransformer", ["P", "S"], "cache"),
}
UMBRAL_CACHE = 0.02
UMBRALES_RESUMEN = [0.05, 0.10, 0.20, 0.30, 0.50]
METAS = [0.80, 0.90, 0.95]
FLUJOS_FIGURA = ["PN-P", "PN-S", "PN-PS", "EQT-D", "EQT-P"]
NOMBRE_FILTRO = {"f1_15": "1-15 Hz", "f1_45": "1-45 Hz"}


def onsets_de(flujo, curvas, umbral, cache):
    """Instantes detectados por un flujo a un umbral."""
    modelo, etiquetas, forma = FLUJOS[flujo]
    partes = []
    for et in etiquetas:
        if forma == "cache":
            # como la cache anterior: disparos a 0,02 y filtro por pico
            clave = (et, UMBRAL_CACHE)
            if clave not in cache:
                cache[clave] = lib.disparos_por_tramo(curvas[et], UMBRAL_CACHE)
            _, _, pico, valor = cache[clave]
            partes.append(pico[valor >= umbral])
        else:
            clave = (et, umbral)
            if clave not in cache:
                cache[clave] = lib.disparos_por_tramo(curvas[et], umbral)
            ini, _, pico, _ = cache[clave]
            partes.append(ini if forma == "inicio" else pico)
    t = np.concatenate(partes) if partes else np.array([], dtype=np.int64)
    if len(etiquetas) > 1:
        t = lib.agrupar_anclado(t, TOL)
    return np.sort(t)


def calcular(eventos, zona_ini, zona_fin, valida):
    inc = eventos[eventos.incluido].reset_index(drop=True)
    tramo_ev = inc["tramo"].to_numpy()
    horas = {"todos": valida.sum() / FS / 3600}
    for k, (a, b) in enumerate(lib.LIMITES_TRAMO, 1):
        horas[k] = valida[a:b].sum() / FS / 3600

    filas_c, filas_t = [], []
    for filtro in lib.FILTROS:
        for modelo in ("phasenet", "eqtransformer"):
            ruta = lib.ruta_anot(modelo, filtro)
            if not os.path.exists(ruta):
                print(f"  (falta {os.path.basename(ruta)}: se omite)")
                continue
            z = np.load(ruta)
            curvas = {k: z[k] for k in z.files}
            cache = {}
            for flujo, (m, ets, _) in FLUJOS.items():
                if m != modelo or any(e not in curvas for e in ets):
                    continue
                for umbral in lib.UMBRALES:
                    o = onsets_de(flujo, curvas, umbral, cache)
                    res = lib.evaluar(o, inc, zona_ini, zona_fin, valida)
                    for tramo in ["todos", 1, 2, 3]:
                        if tramo == "todos":
                            me, mo = None, None
                        else:
                            a, b = lib.LIMITES_TRAMO[tramo - 1]
                            me = tramo_ev == tramo
                            mo = (res["onsets"] >= a) & (res["onsets"] < b)
                        fc, ft = lib.resumir(res, inc, horas[tramo], me, mo)
                        base = dict(flujo=flujo, filtro=filtro, umbral=umbral, tramo=tramo)
                        filas_c += [dict(base, **f) for f in fc]
                        filas_t.append(dict(base, **ft))
            print(f"  {modelo}/{filtro}: listo", flush=True)
    return pd.DataFrame(filas_c), pd.DataFrame(filas_t)


def tabla_resumen(sc, st, flujo, filtro):
    c = sc[(sc.flujo == flujo) & (sc.filtro == filtro) & (sc.tramo == "todos")]
    t = st[(st.flujo == flujo) & (st.filtro == filtro) & (st.tramo == "todos")]
    if c.empty:
        return
    print(f"\n{flujo} | filtro {NOMBRE_FILTRO[filtro]}")
    print(f"  {'umbral':>6} {'VT':>7} {'LP':>7} {'TR':>7} {'VOLC':>9} {'AV':>5} {'IC':>7}"
          f" {'FP/h':>7} {'FDR':>6} {'FDRest':>7}")
    for u in UMBRALES_RESUMEN:
        cc = c[np.isclose(c.umbral, u)].set_index("clase")
        tt = t[np.isclose(t.umbral, u)]
        if cc.empty:
            continue
        kn = lambda k: f"{int(cc.loc[k, 'k'])}/{int(cc.loc[k, 'n'])}"
        tt = tt.iloc[0]
        print(f"  {u:>6.2f} {kn('VT'):>7} {kn('LP'):>7} {kn('TR'):>7} "
              f"{cc.loc['volcanico', 'sens']:>9.3f} {kn('AV'):>5} {kn('IC'):>7}"
              f" {tt.fp_h:>7.1f} {tt.fdr:>6.3f} {tt.fdr_estricto:>7.3f}")


def metas(sc, st):
    v = sc[(sc.clase == "volcanico") & (sc.tramo == "todos")].merge(
        st[st.tramo == "todos"], on=["flujo", "filtro", "umbral", "tramo"])
    v = v[~v.flujo.str.endswith("-ant")]
    print("\nCONFIGURACION CON MENOS FP/h QUE ALCANZA CADA META (sensibilidad volcanica, "
          "nivel 1, traza continua de FRE)")
    print(f"  {'meta':>5} {'flujo':>7} {'filtro':>8} {'umbral':>6} {'sens':>6} "
          f"{'IC95':>13} {'FP/h':>7} {'FDR':>6}")
    for meta in METAS:
        ok = v[v.sens >= meta]
        if ok.empty:
            mx = v.loc[v.sens.idxmax()]
            print(f"  {meta:>5.2f}  no se alcanza. Maximo {mx.sens:.3f} ({mx.flujo}, "
                  f"{NOMBRE_FILTRO[mx.filtro]}, umbral {mx.umbral:.2f}, {mx.fp_h:.1f} FP/h)")
            continue
        # menor FP/h; ante empate, el umbral mas alto
        r = ok.sort_values(["fp_h", "umbral"], ascending=[True, False]).iloc[0]
        print(f"  {meta:>5.2f} {r.flujo:>7} {NOMBRE_FILTRO[r.filtro]:>8} {r.umbral:>6.2f} "
              f"{r.sens:>6.3f} [{r.ic_inf:.3f},{r.ic_sup:.3f}] {r.fp_h:>7.1f} {r.fdr:>6.3f}")
    print("\nPor flujo y filtro (umbral mas alto que alcanza la meta; '-' si no la alcanza):")
    print(f"  {'flujo':>7} {'filtro':>8} {'sens max':>9} " +
          " ".join(f"{'meta ' + format(m, '.2f'):>22}" for m in METAS))
    for (fl, fi), g in v.groupby(["flujo", "filtro"]):
        celdas = []
        for meta in METAS:
            ok = g[g.sens >= meta]
            if ok.empty:
                celdas.append(f"{'-':>22}")
            else:
                r = ok.loc[ok.umbral.idxmax()]
                celdas.append(f"{'u=' + format(r.umbral, '.2f') + ' ' + format(r.fp_h, '.1f') + ' FP/h':>22}")
        print(f"  {fl:>7} {NOMBRE_FILTRO[fi]:>8} {g.sens.max():>9.3f} " + " ".join(celdas))
    print("\nLECTURA COMO PRECISION: umbral mas bajo con FDR <= 0,05 (precision >= 0,95)")
    print(f"  {'flujo':>7} {'filtro':>8} {'umbral':>6} {'sens volc':>9} {'FP/h':>7} {'FDR':>6}")
    for (fl, fi), g in v.groupby(["flujo", "filtro"]):
        ok = g[g.fdr <= 0.05]
        if ok.empty:
            print(f"  {fl:>7} {NOMBRE_FILTRO[fi]:>8}  no se alcanza (FDR minimo {g.fdr.min():.3f})")
            continue
        r = ok.loc[ok.umbral.idxmin()]
        print(f"  {fl:>7} {NOMBRE_FILTRO[fi]:>8} {r.umbral:>6.2f} {r.sens:>9.3f} {r.fp_h:>7.1f} {r.fdr:>6.3f}")


def auditoria(sc):
    print("\nAUDITORIA: extraccion de arribos como la cache anterior (umbral 0,02 y filtro"
          " por confianza) contra la extraccion exacta, sobre los MISMOS datos")
    print(f"  {'modelo':>6} {'filtro':>8} {'umbral':>6} {'volc exacta':>12} {'volc cache':>11}"
          f" {'5 clases exacta':>16} {'5 clases cache':>15}")
    for base, ant in (("PN-PS", "PN-PS-ant"), ("EQT-PS", "EQT-PS-ant")):
        for filtro in lib.FILTROS:
            for u in UMBRALES_RESUMEN:
                def val(fl, clase):
                    r = sc[(sc.flujo == fl) & (sc.filtro == filtro) & np.isclose(sc.umbral, u)
                           & (sc.tramo == "todos") & (sc.clase == clase)]
                    return r.sens.iloc[0] if len(r) else float("nan")
                a, b = val(base, "volcanico"), val(ant, "volcanico")
                c, d = val(base, "todas"), val(ant, "todas")
                if np.isnan(a):
                    continue
                print(f"  {base:>6} {NOMBRE_FILTRO[filtro]:>8} {u:>6.2f} {a:>12.3f} {b:>11.3f}"
                      f" {c:>16.3f} {d:>15.3f}")


def desfases(sc):
    print("\nDESFASE MEDIANO deteccion - inicio catalogado (s), umbral 0,30, aciertos:")
    print(f"  {'flujo':>7} {'filtro':>8} " + " ".join(f"{c:>6}" for c in lib.CLASES))
    for (fl, fi), g in sc[np.isclose(sc.umbral, 0.30) & (sc.tramo == "todos")
                          & ~sc.flujo.str.endswith("-ant")].groupby(["flujo", "filtro"]):
        g = g.set_index("clase")
        print(f"  {fl:>7} {NOMBRE_FILTRO[fi]:>8} " +
              " ".join(f"{g.loc[c, 'dt_mediana_s']:>6.2f}" for c in lib.CLASES))


def figuras(sc, st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Paleta categorica validada (slots 1-3) y tinta para total y FDR
    color = {"VT": "#2a78d6", "LP": "#eb6834", "TR": "#1baf7a"}
    tinta, tinta2, rejilla = "#0b0b0b", "#52514e", "#e4e3df"

    def panel(ax, flujo, filtro):
        c = sc[(sc.flujo == flujo) & (sc.filtro == filtro) & (sc.tramo == "todos")]
        t = st[(st.flujo == flujo) & (st.filtro == filtro) & (st.tramo == "todos")].sort_values("umbral")
        if c.empty:
            ax.set_visible(False)
            return
        lineas = []
        for cl in ["VT", "LP", "TR"]:
            g = c[c.clase == cl].sort_values("umbral")
            lineas.append((cl, g.umbral, g.sens, dict(color=color[cl], lw=2)))
        g = c[c.clase == "volcanico"].sort_values("umbral")
        lineas.append(("Volcanico", g.umbral, g.sens, dict(color=tinta, lw=3)))
        lineas.append(("FDR", t.umbral, t.fdr, dict(color=tinta2, lw=2, ls="--")))
        # referencia de la meta de sensibilidad
        ax.axhline(0.95, color=tinta2, lw=0.8, ls=":", zorder=0)
        ax.text(0.99, 0.95, "0,95", ha="right", va="bottom", fontsize=7, color=tinta2,
                transform=ax.get_yaxis_transform())
        for nombre, x, y, kw in lineas:
            ax.plot(x, y, label=nombre, **kw)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{flujo}, filtro {NOMBRE_FILTRO[filtro]}", fontsize=10, color=tinta, loc="left")
        ax.grid(axis="y", color=rejilla, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(tinta2)
        ax.tick_params(colors=tinta2, labelsize=8)

    filas = [f for f in FLUJOS_FIGURA if f in set(sc.flujo)]
    if not filas:
        return
    fig, axs = plt.subplots(len(filas), 2, figsize=(10, 2.9 * len(filas)),
                            sharex=True, sharey=True, squeeze=False)
    for i, fl in enumerate(filas):
        for j, fi in enumerate(lib.FILTROS):
            panel(axs[i, j], fl, fi)
    for ax in axs[-1]:
        ax.set_xlabel("umbral de confianza", color=tinta2, fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("sensibilidad / FDR", color=tinta2, fontsize=9)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.suptitle("Sensibilidad por clase y FDR por umbral (nivel 1, FRE 3 componentes)",
                 y=0.998, fontsize=11, color=tinta)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    ruta = os.path.join(lib.DIR_SALIDA, "curvas_sens_fdr.png")
    fig.savefig(ruta, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"\nFigura: {ruta}")

    for fl in filas:
        for fi in lib.FILTROS:
            if sc[(sc.flujo == fl) & (sc.filtro == fi)].empty:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 4.4))
            panel(ax, fl, fi)
            ax.set_xlabel("umbral de confianza", color=tinta2, fontsize=9)
            ax.set_ylabel("sensibilidad / FDR", color=tinta2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="upper center",
                      bbox_to_anchor=(0.5, -0.16), ncol=5)
            fig.tight_layout()
            fig.savefig(os.path.join(lib.DIR_SALIDA, f"curva_{fl}_{fi}.png"), dpi=150,
                        facecolor="#fcfcfb")
            plt.close(fig)


def main():
    if not os.path.exists(lib.RUTA_EVENTOS):
        raise SystemExit("Falta eventos_fre3c.csv: correr antes preparar_fre3c.py")
    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    valida = np.load(lib.ruta_datos("f1_15"))["valida"]

    # Zonas con senal real: todos los eventos que el crudo si contiene
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zona_ini = (con_senal.ini - TOL).to_numpy()
    zona_fin = con_senal.fin.to_numpy()

    print("Calculando ...")
    sc, st = calcular(eventos, zona_ini, zona_fin, valida)
    sc.to_csv(os.path.join(lib.DIR_SALIDA, "sens_clase.csv"), index=False)
    st.to_csv(os.path.join(lib.DIR_SALIDA, "sens_totales.csv"), index=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        inc = eventos[eventos.incluido]
        print("EVENTOS EVALUADOS (incluidos) POR CLASE")
        print("  " + "  ".join(f"{c}={int((inc.clase == c).sum())}" for c in lib.CLASES) +
              f"  | volcanicos={int(inc.clase.isin(lib.VOLCANICAS).sum())}"
              f"  | horas validas={valida.sum() / FS / 3600:.2f}")
        excl = eventos[~eventos.incluido]
        if len(excl):
            print("  excluidos: " + ", ".join(
                f"{m} {g.clase.value_counts().to_dict()}" for m, g in excl.groupby("motivo_exclusion")))
        print("\nNIVEL 1 (deteccion del inicio, tolerancia 5 s) sobre la traza continua de FRE."
              "\nk/n por clase; VOLC = sensibilidad VT+LP+TR; FP/h = detecciones fuera de todo"
              " evento; FDR = FP/(aciertos volcanicos + FP); FDRest = 1 - aciertos volcanicos/"
              "detecciones.")
        for fl in FLUJOS:
            if fl.endswith("-ant"):
                continue
            for fi in lib.FILTROS:
                tabla_resumen(sc, st, fl, fi)
        metas(sc, st)
        auditoria(sc)
        desfases(sc)
    texto = buf.getvalue()
    print(texto)
    with open(os.path.join(lib.DIR_SALIDA, "resumen_fre3c.txt"), "w") as f:
        f.write(texto)
    figuras(sc, st)
    print(f"\nTablas en {lib.DIR_SALIDA}/sens_clase.csv y sens_totales.csv")


if __name__ == "__main__":
    main()
