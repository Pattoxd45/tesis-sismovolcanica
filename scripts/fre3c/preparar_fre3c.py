#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preparar_fre3c.py
=================

Prepara la estacion FRE con sus tres componentes reales (Z, N, E) a partir
de los datos crudos (datos/crudos/FRE_*.npy), sobre la grilla de muestras
de la traza de Zenodo, para que el catalogo de referencia se use tal cual.

Pasos:
  1. Alinea cada componente cruda a la grilla de Zenodo con a_grilla() de
     verificar_crudos.py (la misma logica ya validada: los huecos se
     rellenan por interpolacion lineal solo para que el filtro no vea
     escalones, y se marcan como no validos con un margen de 20 s).
  2. Mide el desfase de cada tramo por correlacion cruzada de Z filtrada
     1-15 Hz contra la fila 1 de Zenodo y lo corrige (esperado: 0, 0 y 1
     muestra). Si el desfase supera 5 muestras se detiene.
  3. Filtra las tres componentes con cada filtro (1-15 y 1-45 Hz;
     Butterworth orden 5, fase cero, sosfiltfilt sobre el tramo completo,
     igual que verificar_crudos.py).
  4. Marca cada evento del catalogo: guarda de union, si la fila 1 de
     Zenodo coincide con el crudo dentro del evento (los AV insertados en
     la traza no estan en el crudo), y si su inicio cae en zona valida.
  5. V1: valida que Z filtrada 1-15 Hz reproduce la fila 1 de Zenodo en
     cada tramo (fuera de los eventos que no coinciden). Si la correlacion
     de algun tramo es menor que 0,99, se detiene.

Salidas en ~/tesis/resultados/fre3c/:
    fre3c_f1_15.npz, fre3c_f1_45.npz   Z, N, E (float32), valida (bool)
    eventos_fre3c.csv                  catalogo con banderas e inclusion

Uso:
    python preparar_fre3c.py

Autor: Patricio Valdes Troncoso - Universidad Catolica de Temuco
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "diagnostico"))
import fre3c_lib as lib            # noqa: E402
import verificar_crudos as vc      # noqa: E402

FS = lib.FS
COMPONENTES = ("Z", "N", "E")
LAG_BUSQUEDA = 50          # +-0,5 s para el desfase por tramo
LAG_MAX_ACEPTADO = 5       # mas que esto es un problema de alineacion
R_MIN_V1 = 0.99            # correlacion minima Z(1-15 Hz) vs fila 1
BORDE = vc.MARGEN_HUECO     # 20 s excluidos en los bordes de cada tramo para V1
# Mismos umbrales que verificar_crudos.py para decidir si un evento coincide
ERR_COINCIDE = vc.ERR_COINCIDE        # 0,05
ERR_NO_COINCIDE = vc.ERR_NO_COINCIDE  # 0,5


def cargar_crudo(nombre):
    ruta = os.path.join(vc.DIR_CRUDOS, nombre)
    return np.load(ruta, allow_pickle=True).item()


def preparar_tramo(nombre, n_tramo, ini, fin, zen, sos15):
    """Alinea las tres componentes de un tramo y mide su desfase."""
    d = cargar_crudo(nombre)
    t0 = float(zen[0, ini])
    n = fin - ini
    crudo, valida = {}, np.ones(n, dtype=bool)
    info = dict(tramo=n_tramo, archivo=nombre, estacion=d.get("estacion", "?"))
    for c in COMPONENTES:
        y, v, n_falt, n_fuera = vc.a_grilla(d[c], d["tiempos_unix"][c], t0, n)
        crudo[c] = y
        valida &= v
        info[f"faltantes_{c}"] = n_falt
        info[f"fuera_{c}"] = n_fuera

    # Desfase de Z filtrada 1-15 Hz contra la fila 1 de Zenodo
    zf = sosfiltfilt(sos15, crudo["Z"])
    z1 = np.asarray(zen[1, ini:fin], dtype=float)
    v1 = z1 != 0
    xm = np.where(valida, zf, 0.0)
    zm = np.where(v1, z1, 0.0)
    lags, cc = vc.xcorr_lags(xm, zm, LAG_BUSQUEDA)
    lag = int(lags[int(np.argmax(np.abs(cc)))])
    info["desfase_muestras"] = lag
    if abs(lag) > LAG_MAX_ACEPTADO:
        raise SystemExit(f"DETENIDO: el tramo {n_tramo} tiene desfase {lag} muestras "
                         f"(> {LAG_MAX_ACEPTADO}). Revisar la alineacion antes de seguir.")
    if lag != 0:
        for c in COMPONENTES:
            crudo[c] = lib.desplazar(crudo[c], lag, 0.0)
        valida = lib.desplazar(valida, lag, False)
    return crudo, valida, info


def marcar_eventos(cat, z15_global, zen, valida_global):
    """
    Compara, evento por evento, Z filtrada 1-15 Hz con la fila 1 de Zenodo,
    sin factor de escala, igual que por_evento() de verificar_crudos.py (el
    procedimiento que encontro los AV insertados). Un factor ajustado sobre el
    tramo completo se sesga con esos mismos AV y ensucia la comparacion.
    """
    guarda = lib.mascara_guardas()
    tol = int(lib.TOL_S * FS)
    filas = []
    for i, r in enumerate(cat.itertuples()):
        a, b = int(r.idx_start), int(r.idx_end)
        tramo = next(k for k, (x, y) in enumerate(lib.LIMITES_TRAMO, 1) if x <= a < y)
        en_guarda = not (guarda[a] and guarda[b - 1])
        z = np.asarray(zen[1, a:b], dtype=float)
        x = z15_global[a:b].astype(float)
        ok = valida_global[a:b] & (z != 0)
        if ok.sum() >= 50:
            rr = float(np.corrcoef(x[ok], z[ok])[0, 1])
            err = float(np.sqrt(np.mean((z[ok] - x[ok]) ** 2)) / np.sqrt(np.mean(z[ok] ** 2)))
            estado = ("coincide" if err < ERR_COINCIDE else
                      "no_coincide" if err > ERR_NO_COINCIDE else "parcial")
        else:
            rr, err, estado = float("nan"), float("nan"), "sin_dato"
        lo, hi = max(0, a - tol), min(lib.N_TOTAL, a + tol + 1)
        inicio_valido = bool(valida_global[lo:hi].all())
        motivo = ("guarda_union" if en_guarda else
                  "sin_senal_en_crudo" if estado == "no_coincide" else
                  "sin_dato_crudo" if estado == "sin_dato" else
                  "inicio_en_hueco" if not inicio_valido else "")
        filas.append(dict(id=i, clase=r.event_type, ini=a, fin=b,
                          dur_s=(b - a) / FS, tramo=tramo, en_guarda=en_guarda,
                          r_crudo=rr, err_crudo=err, estado_crudo=estado,
                          frac_valida=float(valida_global[a:b].mean()),
                          inicio_valido=inicio_valido,
                          incluido=motivo == "", motivo_exclusion=motivo))
    return pd.DataFrame(filas)


def main():
    os.makedirs(lib.DIR_SALIDA, exist_ok=True)
    print("Validacion del filtro 1-15 Hz (caso sintetico de verificar_crudos):")
    if not vc.validar_filtro():
        raise SystemExit("DETENIDO: el filtro no pasa la validacion sintetica.")

    zen = np.load(lib.NPY_ZENODO, mmap_mode="r")
    if zen.shape[-1] != lib.N_TOTAL:
        raise SystemExit(f"DETENIDO: la traza tiene {zen.shape[-1]} muestras, "
                         f"se esperaban {lib.N_TOTAL}.")
    # Cortes leidos desde la fila 0, no supuestos
    saltos = np.where(np.abs(np.diff(np.asarray(zen[0])) - 1 / FS) > 1.0)[0] + 1
    print(f"\nCortes leidos de la fila 0: {saltos.tolist()} (esperados {lib.CORTES})")
    if saltos.tolist() != lib.CORTES:
        raise SystemExit("DETENIDO: los cortes no coinciden con los esperados.")
    tramos_vc = [(t[0], t[1], t[2], t[3]) for t in vc.TRAMOS]
    if [(t[2], t[3]) for t in tramos_vc] != lib.LIMITES_TRAMO:
        raise SystemExit("DETENIDO: los tramos de verificar_crudos.py no coinciden con fre3c_lib.")

    sos15 = lib.sos_filtro(lib.FILTROS["f1_15"])
    crudo_g = {c: np.zeros(lib.N_TOTAL) for c in COMPONENTES}
    valida_g = np.zeros(lib.N_TOTAL, dtype=bool)
    infos = []
    for nombre, n_tramo, ini, fin in tramos_vc:
        print(f"\nTramo {n_tramo}: {nombre}")
        crudo, valida, info = preparar_tramo(nombre, n_tramo, ini, fin, zen, sos15)
        for c in COMPONENTES:
            crudo_g[c][ini:fin] = crudo[c]
        valida_g[ini:fin] = valida
        info["horas_validas"] = valida.sum() / FS / 3600
        infos.append(info)
        print("  " + ", ".join(f"{k}={v}" for k, v in info.items() if k not in ("archivo",)))

    valida_g &= lib.mascara_guardas()

    # Filtrado por tramo para cada filtro
    filtrados = {}
    for nombre_f, banda in lib.FILTROS.items():
        sos = lib.sos_filtro(banda)
        filtrados[nombre_f] = {}
        for c in COMPONENTES:
            y = np.zeros(lib.N_TOTAL, dtype=np.float32)
            for a, b in lib.LIMITES_TRAMO:
                y[a:b] = sosfiltfilt(sos, crudo_g[c][a:b]).astype(np.float32)
            filtrados[nombre_f][c] = y

    # Sanidad de N y E: no deben ser copias de Z ni ceros
    print("\nSanidad de componentes (f1_15, muestras validas):")
    z15 = filtrados["f1_15"]
    for k, (a, b) in enumerate(lib.LIMITES_TRAMO, 1):
        m = valida_g[a:b]
        rms = {c: float(np.sqrt(np.mean(z15[c][a:b][m].astype(float) ** 2))) for c in COMPONENTES}
        rzn = np.corrcoef(z15["Z"][a:b][m], z15["N"][a:b][m])[0, 1]
        rze = np.corrcoef(z15["Z"][a:b][m], z15["E"][a:b][m])[0, 1]
        print(f"  tramo {k}: RMS Z={rms['Z']:.1f} N={rms['N']:.1f} E={rms['E']:.1f} | "
              f"r(Z,N)={rzn:.3f} r(Z,E)={rze:.3f}")
        if min(rms.values()) == 0 or max(abs(rzn), abs(rze)) > 0.99:
            raise SystemExit("DETENIDO: alguna componente es cero o copia de Z.")

    cat = pd.read_csv(lib.CSV_CATALOGO)
    ev = marcar_eventos(cat, z15["Z"], zen, valida_g)

    # V1: correlacion por tramo fuera de los eventos que no coinciden
    print("\nV1. Z filtrada 1-15 Hz contra la fila 1 de Zenodo "
          "(fuera de los eventos que no coinciden):")
    fuera = np.ones(lib.N_TOTAL, dtype=bool)
    for r in ev[ev.estado_crudo == "no_coincide"].itertuples():
        fuera[r.ini:r.fin] = False
    v1_ok = True
    for k, (a, b) in enumerate(lib.LIMITES_TRAMO, 1):
        z1 = np.asarray(zen[1, a:b], dtype=float)
        x = z15["Z"][a:b].astype(float)
        m = valida_g[a:b] & (z1 != 0) & fuera[a:b]
        # se excluyen los transitorios del filtro en los bordes del tramo
        m[:BORDE] = False
        m[-BORDE:] = False
        rr = float(np.corrcoef(x[m], z1[m])[0, 1])
        escala = float(np.dot(x[m], z1[m]) / np.dot(x[m], x[m]))
        err = float(np.sqrt(np.mean((z1[m] - escala * x[m]) ** 2)) / np.sqrt(np.mean(z1[m] ** 2)))
        ok = rr >= R_MIN_V1
        v1_ok &= ok
        print(f"  tramo {k}: r={rr:.5f} escala={escala:.5g} err_rel={err:.5f} "
              f"desfase={infos[k - 1]['desfase_muestras']} -> {'PASA' if ok else 'FALLA'}")
    if not v1_ok:
        raise SystemExit("DETENIDO: V1 no pasa. No se guardan salidas.")

    # Eventos
    print("\nEventos por clase (incluidos / total) y motivos de exclusion:")
    print(pd.crosstab(ev.clase, ev.motivo_exclusion.replace("", "incluido"),
                      margins=True).to_string())
    raros = ev[(ev.estado_crudo != "coincide") & (ev.clase != "AV")]
    if len(raros):
        print("\nATENCION: eventos que no son AV y no coinciden por completo con el crudo:")
        print(raros[["id", "clase", "tramo", "ini", "dur_s", "r_crudo", "err_crudo",
                     "estado_crudo", "incluido"]].to_string(index=False))
    av = ev[(ev.clase == "AV") & (ev.estado_crudo == "no_coincide")]
    print(f"\nAV sin senal en el crudo: {len(av)} (esperado 17, todos del tramo 1; "
          f"tramos: {sorted(av.tramo.unique().tolist())})")

    # Guardado
    for nombre_f, comps in filtrados.items():
        np.savez(lib.ruta_datos(nombre_f), Z=comps["Z"], N=comps["N"], E=comps["E"],
                 valida=valida_g, banda=np.array(lib.FILTROS[nombre_f]),
                 desfases=np.array([i["desfase_muestras"] for i in infos]))
        print(f"Guardado {lib.ruta_datos(nombre_f)}")
    ev.to_csv(lib.RUTA_EVENTOS, index=False)
    pd.DataFrame(infos).to_csv(os.path.join(lib.DIR_SALIDA, "tramos_fre3c.csv"), index=False)
    print(f"Guardado {lib.RUTA_EVENTOS}")
    print(f"Horas validas totales: {valida_g.sum() / FS / 3600:.3f}")


if __name__ == "__main__":
    main()
