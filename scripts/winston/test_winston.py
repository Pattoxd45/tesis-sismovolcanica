#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para winston_lib.py: decodificacion de paquetes TRACEBUF2 y armado de la grilla.

Uso: python test_winston.py
"""

import os
import struct
import sys
import zlib
from datetime import date, datetime, timezone

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import cobertura_jose as cj
import extraer_ventanas_jose as evj
import extraer_winston as ew
import winston_lib as wl

# 2018-02-09 00:00 UTC, el dia con que se valido la lectura
T0 = 1518134400.0


def paquete(t_ini, muestras, tipo="s4", sr=100.0, comprimir=True, recortar=0):
    orden = ">" if tipo[0] == "s" else "<"
    n = len(muestras)
    cab = struct.pack(orden + "iiddd", 7, n, t_ini, t_ini + (n - 1) / sr, sr)
    for s, ancho in (("FREZ", 7), ("TC", 9), ("HHZ", 4), ("99", 3), ("20", 2), (tipo, 3), ("", 2), ("", 2)):
        cab += s.encode().ljust(ancho, b"\0")
    buf = cab + np.asarray(muestras, dtype=f"{orden}i{tipo[1]}").tobytes()
    buf = buf[:len(buf) - recortar]
    return zlib.compress(buf) if comprimir else buf


def fila(blob, st=0.0):
    return (st, st + 1, 100.0, "s4", blob)


def caso(nombre, ok):
    print(f"  {'OK   ' if ok else 'FALLA'} {nombre}")
    return ok


def main():
    todo = True
    m = [5, -3, 2147483647, -2147483648, 0]

    cab, x = wl.decodificar(paquete(T0, m, "s4"))
    todo &= caso("s4 comprimido: muestras y cabecera",
                 x.tolist() == m and cab["sta"] == "FREZ" and cab["chan"] == "HHZ" and cab["loc"] == "99"
                 and cab["comprimido"] and cab["nsamp"] == 5)

    cab, x = wl.decodificar(paquete(T0, m, "i4", comprimir=False))
    todo &= caso("i4 sin comprimir", x.tolist() == m and not cab["comprimido"])

    cab, x = wl.decodificar(paquete(T0, [1, -2, 32767], "s2"))
    todo &= caso("s2 (16 bits)", x.tolist() == [1, -2, 32767])

    try:
        wl.decodificar(paquete(T0, m, "s4", recortar=4))
        todo &= caso("largo que no calza da error", False)
    except ValueError:
        todo &= caso("largo que no calza da error", True)

    a = list(range(100))
    b = list(range(100, 200))
    x, info = wl.armar([fila(paquete(T0, a)), fila(paquete(T0 + 2.0, b))], T0, 400)
    todo &= caso("dos paquetes con un hueco de 1 s",
                 x[:100].tolist() == a and np.isnan(x[100:200]).all() and x[200:300].tolist() == b
                 and np.isnan(x[300:]).all() and info["validas"] == 200)

    x, info = wl.armar([fila(paquete(T0 + 1.0 - wl.J2K_A_UNIX, a))], T0, 300)
    todo &= caso("hora en J2K se convierte", info["j2k"] == 1 and x[100:200].tolist() == a)

    x, info = wl.armar([fila(paquete(T0, a)), fila(paquete(T0 + 0.5, [999] * 100))], T0, 300)
    todo &= caso("paquetes que se pisan con otro valor se cuentan", info["conflictos"] == 50)

    x, info = wl.armar([fila(paquete(T0, a, sr=50.0))], T0, 300)
    todo &= caso("otra frecuencia se descarta", info["otra_fs"] == 1 and info["validas"] == 0)

    x, info = wl.armar([fila(paquete(T0 + 0.004, a))], T0, 300)
    todo &= caso("desvio del inicio a la grilla", abs(info["desvio_max"] - 0.4) < 1e-4 and x[0] == 0)

    blob = paquete(T0, a)
    salida = f"568080000.5\t568080001.5\t100\ts4\t{blob.hex().upper()}\n1\t2\t100\ts4\tNULL\n"
    filas, nulas = wl.parsear_salida(salida)
    todo &= caso("lectura de la salida de mysql", len(filas) == 1 and nulas == 1 and filas[0][4] == blob)

    rng = np.random.default_rng(0)
    s = rng.integers(-1000, 1000, 20000).astype(float)
    lag, frac = wl.mejor_desfase(s, np.roll(s, 37), 100, largo=5000)
    todo &= caso("desfase conocido se encuentra", lag == 37 and frac == 1.0)

    sql = wl.sql_tabla("winston", "fre_z_2018_02_09", 1.5, 2.5)
    todo &= caso("consulta con filtro por hora J2K",
                 sql.endswith("WHERE st >= 1.500 AND st < 2.500 ORDER BY st")
                 and "WHERE" not in wl.sql_tabla("winston", "t"))

    u = lambda s_: ew.utc(s_)
    d9, d10 = date(2018, 2, 9), date(2018, 2, 10)
    todo &= caso("dias de una ventana que cruza la medianoche",
                 ew.dias(u("2018-02-09T23:59:00"), u("2018-02-10T00:10:00")) == [d9, d10])
    todo &= caso("ventana recien pasada la medianoche incluye el dia anterior",
                 ew.dias(u("2018-02-10T00:00:05"), u("2018-02-10T00:05:00")) == [d9, d10])
    todo &= caso("ventana lejos de la medianoche usa un solo dia",
                 ew.dias(u("2018-02-10T00:01:00"), u("2018-02-10T00:05:00")) == [d10])
    todo &= caso("hora sin zona se toma como UTC",
                 u("2018-02-09T00:00:00") == datetime(2018, 2, 9, tzinfo=timezone.utc))
    todo &= caso("nombres en el respaldo y en la base local",
                 ew.patron_nas("w_free$hhe$tc$99", d9) == "free$hhe$tc$99$$2018_02_09"
                 and ew.patron_nas("W_FREZ$HHZ$TC$99", d9) == "FREZ$HHZ$TC$99$$2018_02_09"
                 and ew.tabla_local("FRE", "Z", d9) == "fre_z_2018_02_09"
                 and ew.nombre_local("W_FREZ$HHZ$TC$99/frez$hhz$tc$99$$2018_02_09.MYD", "fre_z_2018_02_09")
                 == "fre_z_2018_02_09.MYD")

    m = [cj.DIA_TABLA.search(n) for n in ("frez$hhz$tc$99$$2018_02_09.MYD", "FREZ$HHZ$TC$99$$2014_04_02.MYD",
                                            "frez$hhz$tc$99$$h2018_02_09.MYD", "frez$hhz$tc$99$$2018_02_09.MYI")]
    todo &= caso("dias de tabla en el listado del NAS (sin helicorder ni indices)",
                 m[0].groups() == ("2018", "02", "09") and m[1] is not None and m[2] is None and m[3] is None)

    import pandas as pd
    j9 = datetime(2018, 2, 9, 12, tzinfo=timezone.utc).timestamp() - wl.J2K_A_UNIX
    j10 = datetime(2018, 2, 9, 23, 58, tzinfo=timezone.utc).timestamp() - wl.J2K_A_UNIX
    v = pd.DataFrame(dict(inicio_ventana_w=[j9, j10], fin_ventana_w=[j9 + 300, j10 + 300]))
    c = cj.cobertura(v, {"Z": {d9, d10}, "N": {d9, d10}, "E": {d9}})
    todo &= caso("ventana que cruza la medianoche necesita los dos dias",
                 c.E.tolist() == [True, False] and c.Z.tolist() == [True, True])

    d1, d2, d3 = date(2018, 2, 9), date(2018, 2, 10), date(2018, 2, 11)
    dv = {"a": [d1], "b": [d1, d2], "c": [d3]}
    g1, t1 = evj.tandas(dv, 1)
    g2, t2 = evj.tandas(dv, 2)
    todo &= caso("cada ventana queda en la tanda donde estan todos sus dias",
                 g1 == [[d1], [d2], [d3]] and t1 == [["a"], ["b"], ["c"]]
                 and g2 == [[d1, d2], [d3]] and t2 == [["a", "b"], ["c"]])

    z = np.random.default_rng(3).integers(-5000, 5000, 1000).astype(float)
    zh = z.copy()
    zh[100:150] = np.nan
    f0, k0 = evj.comparar_z(zh, z)
    f1, k1 = evj.comparar_z(z, np.roll(z, -1))
    todo &= caso("Z igual a la de Jose, sin contar los huecos, y desfase de una muestra",
                 f0 == 1.0 and k0 == 0 and f1 == 1.0 and k1 == 1
                 and np.isnan(evj.comparar_z(z, None)[0]))

    print("TODOS PASAN" if todo else "HAY FALLAS")
    sys.exit(0 if todo else 1)


if __name__ == "__main__":
    main()
