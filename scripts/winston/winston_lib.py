#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lectura de tablas Winston copiadas a la MariaDB local: decodifica cada paquete (TRACEBUF2,
comprimido o no) y arma la senal continua en una grilla de 100 Hz.

Uso: se importa desde los otros scripts de scripts/winston.
"""

import getpass
import subprocess
import struct
import zlib

import numpy as np

# frecuencia de los paquetes de FRE en Winston (medida: 100 Hz)
FS = 100.0
# cabecera TRACEBUF2 de Earthworm: 64 bytes antes de las muestras
TAM_CABECERA = 64
# segundos entre 1970-01-01 y 2000-01-01 12:00 UTC (J2K de Winston)
J2K_A_UNIX = 946728000.0
# en esta base las fechas en segundos desde 1970 van de 1,35e9 a 1,6e9; en J2K serian 4e8 a 6,5e8
LIMITE_J2K = 1e9


def texto(buf, a, b):
    return buf[a:b].split(b"\0")[0].decode("ascii", "replace").strip()


def decodificar(blob):
    """Devuelve (cabecera, muestras) de un paquete TRACEBUF2; acepta el paquete comprimido con zlib o no."""
    try:
        buf = zlib.decompress(blob)
        comprimido = True
    except zlib.error:
        buf = bytes(blob)
        comprimido = False
    if len(buf) < TAM_CABECERA:
        raise ValueError(f"paquete de {len(buf)} bytes, menor que la cabecera")
    tipo = texto(buf, 57, 60)
    if len(tipo) != 2 or tipo[0] not in "si" or tipo[1] not in "24":
        raise ValueError(f"tipo de dato no soportado: {tipo!r}")
    # s = orden de Sun (big-endian), i = orden de Intel (little-endian)
    orden = ">" if tipo[0] == "s" else "<"
    tam = int(tipo[1])
    pinno, nsamp, t_ini, t_fin, sr = struct.unpack(orden + "iiddd", buf[:32])
    esperado = TAM_CABECERA + nsamp * tam
    if nsamp <= 0 or len(buf) != esperado:
        raise ValueError(f"largo {len(buf)} distinto de cabecera + {nsamp} muestras ({esperado})")
    cab = dict(pinno=pinno, nsamp=nsamp, t_ini=t_ini, t_fin=t_fin, sr=sr, sta=texto(buf, 32, 39),
               net=texto(buf, 39, 48), chan=texto(buf, 48, 52), loc=texto(buf, 52, 55), tipo=tipo,
               comprimido=comprimido)
    muestras = np.frombuffer(buf, dtype=f"{orden}i{tam}", count=nsamp, offset=TAM_CABECERA)
    return cab, muestras.astype(np.float64)


def parsear_salida(salida):
    """Filas (st, et, sr, datatype, bytes) de la salida de mysql --batch con el paquete en HEX."""
    filas, nulas = [], 0
    for linea in salida.splitlines():
        if not linea.strip():
            continue
        st, et, sr, dt, hx = linea.split("\t")
        if hx == "NULL":
            nulas += 1
            continue
        filas.append((float(st), float(et), float(sr), dt, bytes.fromhex(hx)))
    return filas, nulas


def leer_tabla(base, tabla, usuario=None):
    """Lee una tabla Winston de la MariaDB local con el cliente mysql (sin contrasena: usuario del sistema)."""
    usuario = usuario or getpass.getuser()
    sql = f"SELECT st, et, sr, datatype, HEX(tracebuf) FROM `{base}`.`{tabla}` ORDER BY st"
    r = subprocess.run(["mysql", "-u", usuario, "--batch", "--raw", "--skip-column-names", "-e", sql],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"DETENIDO: mysql fallo con {base}.{tabla}: {r.stderr.strip()}")
    return parsear_salida(r.stdout)


def armar(filas, t0, n, fs=FS):
    """Pone las muestras de cada paquete en una grilla de n muestras desde t0 (NaN donde no hay datos)."""
    x = np.full(n, np.nan)
    info = dict(paquetes=len(filas), malos=0, otra_fs=0, j2k=0, conflictos=0, desvio_max=0.0)
    for _, _, _, _, blob in filas:
        try:
            cab, m = decodificar(blob)
        except ValueError:
            info["malos"] += 1
            continue
        if abs(cab["sr"] - fs) > 1e-6:
            info["otra_fs"] += 1
            continue
        t = cab["t_ini"]
        if t < LIMITE_J2K:
            info["j2k"] += 1
            t += J2K_A_UNIX
        pos = (t - t0) * fs
        i0 = int(round(pos))
        # cuanto se aleja el inicio del paquete de la grilla, en fracciones de muestra
        info["desvio_max"] = max(info["desvio_max"], abs(pos - i0))
        a, b = max(i0, 0), min(i0 + m.size, n)
        if a >= b:
            continue
        seg = m[a - i0:b - i0]
        previo = x[a:b]
        info["conflictos"] += int((~np.isnan(previo) & (previo != seg)).sum())
        x[a:b] = seg
    info["validas"] = int(np.isfinite(x).sum())
    return x, info


def tramos_con_datos(valida):
    """Inicio y fin (exclusivo) de cada tramo seguido de muestras validas."""
    v = np.concatenate([[False], valida, [False]]).astype(np.int8)
    d = np.diff(v)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def mejor_desfase(a, b, lag_max, largo=12000):
    """Desfase de b (en muestras) que da mas muestras iguales con a, en el primer bloque con datos en ambos."""
    ini, fin = tramos_con_datos(np.isfinite(a) & np.isfinite(b))
    necesario = largo + 2 * lag_max
    for i, f in zip(ini, fin):
        if f - i >= necesario:
            s = i + lag_max
            ref = a[s:s + largo]
            mejor = max(((float(np.mean(ref == b[s + k:s + k + largo])), k)
                         for k in range(-lag_max, lag_max + 1)), key=lambda p: (p[0], -abs(p[1])))
            return mejor[1], mejor[0]
    return None, None
