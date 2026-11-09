#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae Z, N y E de una estacion entre dos instantes UTC desde el respaldo Winston del NAS: copia
por SSH solo los dias que faltan (en el NAS solo se lee), los carga en la MariaDB local y guarda
la senal a 100 Hz en un npz.

Uso: TESIS_NAS=usuario@ip python extraer_winston.py --inicio 2018-02-09T04:00:00 --fin 2018-02-09T05:00:00
     [--estacion FRE] [--salida archivo.npz]
"""

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import winston_lib as wl

FS = wl.FS
# respaldo de Winston en el NAS; solo se lee
DIR_NAS = "/mnt/HD/HD_a2/uct/Respaldo para ordenar/Servidor_Cesar/data"
# carpeta de cada componente en el respaldo; las mayusculas cambian entre componentes
CARPETAS = {"FRE": {"Z": "W_FREZ$HHZ$TC$99", "N": "W_FREN$HHN$TC$99", "E": "w_free$hhe$tc$99"}}
# base de la MariaDB local donde quedan las copias de las tablas
BASE_LOCAL = "winston"
DIR_SALIDA = os.path.expanduser("~/tesis/datos/nas/extraidos")
DIR_TMP = os.path.expanduser("~/tesis/datos/nas")
# cada paquete queda en la tabla del dia en que empieza: un aviso pasada la medianoche puede estar en el dia anterior
ANTES_S = 10.0
# una sola conexion SSH para todas las llamadas: la contrasena del NAS se pide una vez
SSH_OPCIONES = ["-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/cm-%r@%h:%p", "-o", "ControlPersist=120"]
# busquedas por llamada, para no pasar el largo maximo de un comando
POR_LLAMADA = 150


def utc(texto):
    t = datetime.fromisoformat(texto)
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def dias(inicio, fin):
    """Dias (UTC) cuyas tablas pueden tener paquetes entre inicio y fin."""
    d, ultimo = (inicio - timedelta(seconds=ANTES_S)).date(), fin.date()
    out = []
    while d <= ultimo:
        out.append(d)
        d += timedelta(days=1)
    return out


def tabla_local(estacion, comp, dia):
    return f"{estacion.lower()}_{comp.lower()}_{dia:%Y_%m_%d}"


def patron_nas(carpeta, dia):
    """Nombre de la tabla del dia en el respaldo, sin extension (se busca sin distinguir mayusculas)."""
    return f"{carpeta[2:]}$${dia:%Y_%m_%d}"


def nombre_local(ruta_nas, tabla):
    return f"{tabla}.{ruta_nas.rsplit('.', 1)[1]}"


def ssh(nas, comando, **kw):
    return subprocess.run(["ssh", *SSH_OPCIONES, nas, comando], **kw)


def mysql_local(sql):
    r = subprocess.run(["mysql", "-u", getpass.getuser(), "--batch", "--skip-column-names", "-e", sql],
                       capture_output=True, text=True, check=False)
    return r.returncode, r.stdout, r.stderr


def tablas_cargadas():
    cod, out, _ = mysql_local(f"SHOW TABLES FROM `{BASE_LOCAL}`")
    return set(out.split()) if cod == 0 else set()


def buscar_en_nas(nas, pedidos):
    """Rutas (carpeta/archivo) que existen en el respaldo para cada (carpeta, patron) pedido."""
    rutas = []
    for i in range(0, len(pedidos), POR_LLAMADA):
        partes = [f"find '{c}' -maxdepth 1 -iname '{p}.*'" for c, p in pedidos[i:i + POR_LLAMADA]]
        r = ssh(nas, f'cd "{DIR_NAS}" && ' + "; ".join(partes), capture_output=True, text=True)
        if r.returncode != 0 and not r.stdout.strip():
            raise SystemExit(f"DETENIDO: no se pudo buscar en el NAS: {r.stderr.strip()}")
        rutas += [l.strip().removeprefix("./") for l in r.stdout.splitlines() if l.strip()]
    return rutas


def copiar_de_nas(nas, nombres, destino):
    """Copia con tar las rutas del respaldo y las deja en destino con su nombre local."""
    rutas = list(nombres)
    copiados = 0
    for i in range(0, len(rutas), POR_LLAMADA):
        cmd = f'cd "{DIR_NAS}" && tar -cf - ' + " ".join(f"'{r}'" for r in rutas[i:i + POR_LLAMADA])
        p = subprocess.Popen(["ssh", *SSH_OPCIONES, nas, cmd], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=p.stdout, mode="r|") as tar:
            for m in tar:
                if m.isfile() and m.name in nombres:
                    with open(os.path.join(destino, nombres[m.name]), "wb") as f:
                        shutil.copyfileobj(tar.extractfile(m), f)
                    copiados += 1
        if p.wait() != 0:
            raise SystemExit("DETENIDO: fallo la copia desde el NAS")
    return copiados


def cargar_en_mariadb(archivos):
    """Pone las copias en la base local; pide la clave de sudo porque la carpeta de MariaDB es de root."""
    _, out, _ = mysql_local("SELECT @@datadir")
    destino = os.path.join(out.strip(), BASE_LOCAL)
    subprocess.run(["sudo", "mysql", "-e", f"CREATE DATABASE IF NOT EXISTS `{BASE_LOCAL}`"], check=True)
    subprocess.run(["sudo", "cp", *archivos, destino + "/"], check=True)
    subprocess.run(["sudo", "chown", "mysql:mysql",
                    *[os.path.join(destino, os.path.basename(a)) for a in archivos]], check=True)
    subprocess.run(["sudo", "mysql", "-e", "FLUSH TABLES"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", required=True, help="UTC, por ejemplo 2018-02-09T04:00:00")
    ap.add_argument("--fin", required=True)
    ap.add_argument("--estacion", default="FRE", choices=sorted(CARPETAS))
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    nas = os.environ.get("TESIS_NAS")
    if not nas:
        raise SystemExit("DETENIDO: falta TESIS_NAS=usuario@ip del NAS")
    inicio, fin = utc(args.inicio), utc(args.fin)
    if fin <= inicio:
        raise SystemExit("DETENIDO: --fin tiene que ser posterior a --inicio")
    est, carpetas = args.estacion, CARPETAS[args.estacion]
    lista_dias = dias(inicio, fin)

    pedidos = {(c, d): tabla_local(est, c, d) for c in carpetas for d in lista_dias}
    ya = tablas_cargadas()
    faltan = {k: t for k, t in pedidos.items() if t not in ya}
    print(f"{est} de {inicio:%Y-%m-%d %H:%M:%S} a {fin:%Y-%m-%d %H:%M:%S} UTC: {len(lista_dias)} dias, "
          f"{len(pedidos)} tablas; ya cargadas {len(pedidos) - len(faltan)}, por traer {len(faltan)}")

    sin_dato = []
    if faltan:
        claves = list(faltan)
        rutas = buscar_en_nas(nas, [(carpetas[c], patron_nas(carpetas[c], d)) for c, d in claves])
        nombres = {}
        for (c, d), tabla in faltan.items():
            pat = (carpetas[c] + "/" + patron_nas(carpetas[c], d)).lower()
            propias = [r for r in rutas if r.lower().rsplit(".", 1)[0] == pat]
            if len(propias) < 3:
                sin_dato.append(tabla)
                continue
            nombres.update({r: nombre_local(r, tabla) for r in propias})
        if nombres:
            os.makedirs(DIR_TMP, exist_ok=True)
            tmp = tempfile.mkdtemp(prefix="tmp_winston_", dir=DIR_TMP)
            try:
                n = copiar_de_nas(nas, nombres, tmp)
                print(f"Copiados {n} archivos del NAS; se cargan en la MariaDB local (puede pedir la clave de sudo)")
                cargar_en_mariadb(sorted(os.path.join(tmp, f) for f in os.listdir(tmp)))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    if sin_dato:
        print(f"Sin tabla en el respaldo (quedan sin datos): {', '.join(sin_dato)}")

    cargadas = tablas_cargadas()
    t0 = inicio.timestamp()
    n = int(round((fin - inicio).total_seconds() * FS))
    # el filtro usa la hora J2K de la tabla; 2 s antes para no perder el paquete que ya venia corriendo
    st_min, st_max = t0 - wl.J2K_A_UNIX - 2.0, fin.timestamp() - wl.J2K_A_UNIX
    senal = {}
    for c in carpetas:
        filas = []
        for d in lista_dias:
            t = tabla_local(est, c, d)
            if t in cargadas:
                filas += wl.leer_tabla(BASE_LOCAL, t, st_min=st_min, st_max=st_max)[0]
        x, info = wl.armar(filas, t0, n)
        senal[c] = x
        ini, _ = wl.tramos_con_datos(np.isfinite(x))
        print(f"  {c}: {info['paquetes']} paquetes ({info['malos']} malos, {info['otra_fs']} con otra frecuencia); "
              f"{info['validas'] / n:.1%} de las muestras con dato, en {len(ini)} tramos; "
              f"muestras que se pisan con otro valor: {info['conflictos']}")

    salida = args.salida or os.path.join(DIR_SALIDA, f"{est}_{inicio:%Y%m%dT%H%M%S}_{fin:%Y%m%dT%H%M%S}.npz")
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    np.savez_compressed(salida, t0=t0, fs=FS, estacion=est, **senal)
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
