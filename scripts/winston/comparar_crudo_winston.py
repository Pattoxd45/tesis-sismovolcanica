#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprueba la lectura de Winston: arma el 9 de febrero de 2018 de FRE (Z, N y E) desde las tablas
copiadas del NAS a la MariaDB local y lo compara con el registro original de FRE del tramo 1.

Uso: python comparar_crudo_winston.py [--base fre] [--guardar]
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import winston_lib as wl

FS = wl.FS
DIR_CRUDOS = os.path.expanduser("~/tesis/datos/crudos")
CRUDO = "FRE_1518149218_1518170818.npy"
TABLAS = {"Z": "z_2018_02_09", "N": "n_2018_02_09", "E": "e_2018_02_09"}
T0 = datetime(2018, 2, 9, tzinfo=timezone.utc).timestamp()
N_DIA = int(86400 * FS)
SALIDA = os.path.expanduser("~/tesis/datos/nas/fre_2018_02_09/fre_2018_02_09.npz")
LAG_MAX = 500


def hora(i):
    return datetime.fromtimestamp(T0 + i / FS, tz=timezone.utc).strftime("%H:%M:%S")


def crudo_en_grilla(d, c):
    y = np.full(N_DIA, np.nan)
    t = np.asarray(d["tiempos_unix"][c], dtype=float)
    pos = (t - T0) * FS
    k = np.round(pos).astype(np.int64)
    ok = (k >= 0) & (k < N_DIA)
    y[k[ok]] = np.asarray(d[c], dtype=float)[ok]
    return y, float(np.max(np.abs(pos - k))) if pos.size else np.nan


def comparar(w, y, lag=0):
    if lag:
        y = np.roll(y, -lag)
    m = np.isfinite(w) & np.isfinite(y)
    n = int(m.sum())
    if n == 0:
        return dict(comunes=0)
    dif = np.abs(w[m] - y[m])
    r = float(np.corrcoef(w[m], y[m])[0, 1]) if n > 1 else np.nan
    return dict(comunes=n, iguales=float(np.mean(dif == 0)), dif_max=float(dif.max()), corr=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="fre")
    ap.add_argument("--guardar", action="store_true", help="guarda el dia armado en un npz")
    args = ap.parse_args()

    print("1. PRIMEROS PAQUETES DE Z (para ver el formato)")
    filas_z, nulas = wl.leer_tabla(args.base, TABLAS["Z"])
    print(f"   {len(filas_z)} filas, {nulas} sin paquete")
    for st, et, sr, dt, blob in filas_z[:3]:
        cab, m = wl.decodificar(blob)
        print(f"   st {st:.3f} | cabecera: inicio {cab['t_ini']:.3f}, {cab['nsamp']} muestras a {cab['sr']} Hz, "
              f"{cab['sta']}.{cab['chan']}.{cab['net']}.{cab['loc']}, tipo {cab['tipo']} (columna {dt}), "
              f"comprimido {cab['comprimido']} | inicio - st = {cab['t_ini'] - st:.3f} s | "
              f"primeras muestras {m[:5].astype(int).tolist()}")

    print("\n2. DIA ARMADO DESDE WINSTON (grilla de 100 Hz desde las 00:00 UTC)")
    dia = {}
    for c, tabla in TABLAS.items():
        filas = filas_z if c == "Z" else wl.leer_tabla(args.base, tabla)[0]
        x, info = wl.armar(filas, T0, N_DIA)
        dia[c] = x
        ini, fin = wl.tramos_con_datos(np.isfinite(x))
        print(f"   {c}: {info['paquetes']} paquetes ({info['malos']} malos, {info['otra_fs']} con otra frecuencia, "
              f"{info['j2k']} con hora J2K); {info['validas'] / FS / 3600:.2f} h con datos en {len(ini)} tramos "
              f"(de {hora(ini[0]) if len(ini) else '-'} a {hora(fin[-1] - 1) if len(fin) else '-'}); "
              f"muestras que se pisan con otro valor: {info['conflictos']}; "
              f"desvio maximo del inicio de paquete: {info['desvio_max']:.3f} muestras")

    print(f"\n3. COMPARACION CON EL REGISTRO ORIGINAL ({CRUDO})")
    ruta = os.path.join(DIR_CRUDOS, CRUDO)
    if not os.path.exists(ruta):
        raise SystemExit(f"DETENIDO: no esta {ruta}")
    d = np.load(ruta, allow_pickle=True).item()
    for c in TABLAS:
        y, desvio = crudo_en_grilla(d, c)
        r0 = comparar(dia[c], y)
        lag, frac = wl.mejor_desfase(dia[c], y, LAG_MAX)
        linea = (f"   {c}: original con {np.isfinite(y).sum() / FS / 3600:.2f} h (desvio de sus tiempos a la grilla "
                 f"{desvio:.3f} muestras); en comun {r0.get('comunes', 0)} muestras")
        if r0.get("comunes"):
            linea += (f"; iguales {r0['iguales']:.4f}, diferencia maxima {r0['dif_max']:.1f}, "
                      f"correlacion {r0['corr']:.6f}")
        print(linea)
        if lag is None:
            print("      sin un bloque seguido de datos para buscar desfase")
        elif lag != 0:
            r1 = comparar(dia[c], y, lag)
            print(f"      el mejor desfase es {lag} muestras ({lag / FS:.2f} s): iguales {r1['iguales']:.4f}, "
                  f"correlacion {r1['corr']:.6f}")
        else:
            print(f"      desfase 0 (el mejor en +-{LAG_MAX} muestras; iguales en el bloque de prueba {frac:.4f})")

    if args.guardar:
        os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
        np.savez_compressed(SALIDA, t0=T0, fs=FS, **dia)
        print(f"\nDia guardado en {SALIDA}")


if __name__ == "__main__":
    main()
