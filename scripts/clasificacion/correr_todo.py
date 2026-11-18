#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corre en orden todas las corridas del clasificador con el protocolo elegido y se salta las que ya tienen
su JSON (si se corta, se vuelve a lanzar igual): EfficientNet-B0 con las dos formas de imagen y ResNet50
y VGG16 con estirar, en los 5 folds de Jose; despues cada red entrenada con todo desarrollo y evaluada
una vez en test, con tantas epocas como la mediana de sus mejores epocas en los folds; al final, el
resumen.

Uso: python correr_todo.py --datos ~/cnn --salida ~/cnn/resultados [--acumular 2] [--acumular-vgg16 4] [--prueba]
     (--datos: carpeta con clasificacion_colab/ y clasificacion_colab_fijo60s/ ya descomprimidas;
      --prueba: una epoca corta de cada red para ver que cabe en la memoria de la GPU, sin guardar nada)
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
ENTRENAR = os.path.join(AQUI, "entrenar_cnn.py")
RESUMEN = os.path.join(AQUI, "resumen_cnn.py")
# protocolo elegido en el fold 1 y confirmado en los folds 2 a 4 con EfficientNet-B0 en Colab
PROTOCOLO = dict(tamano="estirar", pesos_clase=True, ajuste_completo=True, lr=1e-4, duracion_minima=5.0)
ARGS_PROTOCOLO = ["--ajuste-completo", "--pesos-clase", "--lr", "0.0001", "--duracion-minima", "5"]
EPOCAS_CV = 10
FOLDS = [1, 2, 3, 4, 5]
ESTIRAR, FIJO = "clasificacion_colab", "clasificacion_colab_fijo60s"
REDES = [("efficientnet_b0", [ESTIRAR, FIJO]), ("resnet50", [ESTIRAR]), ("vgg16", [ESTIRAR])]


def nombre(modelo, fold, carpeta):
    """Igual que nombre_resultado de entrenar_cnn.py (se repite para no cargar PyTorch aqui)."""
    conjunto = carpeta[len(ESTIRAR):].strip("_")
    return (f"{modelo}_{'test' if fold == 0 else f'fold{fold}'}_estirar" + (f"_{conjunto}" if conjunto else "")
            + "_pesos_completo_lr0.0001_min5s")


def trabajos_cv():
    return [(m, f, c) for m, cs in REDES for f in FOLDS for c in cs]


def epocas_final(salida, modelo):
    """Mediana de las mejores epocas de la red en los 5 folds (con estirar)."""
    ep = []
    for f in FOLDS:
        with open(os.path.join(salida, nombre(modelo, f, ESTIRAR) + ".json")) as a:
            ep.append(json.load(a)["mejor_epoca"])
    return int(round(statistics.median(ep))), ep


def correr(cmd, etiqueta, i, n):
    t0 = time.time()
    print(f"\n=== [{i}/{n}] {etiqueta} ({time.strftime('%H:%M')}) ===", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"DETENIDO: fallo {etiqueta} (codigo {r.returncode}); revisar el mensaje de arriba")
    print(f"=== {etiqueta}: {(time.time() - t0) / 60:.1f} min ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--salida", required=True)
    # lote efectivo de 64 como en Colab; --acumular lo pasa en partes para que quepa en GPU con menos memoria
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--acumular", type=int, default=1)
    ap.add_argument("--acumular-vgg16", type=int, default=None, help="por defecto, el mismo --acumular")
    ap.add_argument("--trabajadores", type=int, default=4)
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    acum = {m: args.acumular for m, _ in REDES}
    acum["vgg16"] = args.acumular_vgg16 or args.acumular
    datos, salida = os.path.expanduser(args.datos), os.path.expanduser(args.salida)
    os.makedirs(salida, exist_ok=True)
    for c in (ESTIRAR, FIJO):
        if not os.path.exists(os.path.join(datos, c, "indice.csv")):
            raise SystemExit(f"DETENIDO: falta {os.path.join(datos, c)} (descomprimir {c}.zip ahi)")
    comun = ["--lote", str(args.lote), "--trabajadores", str(args.trabajadores), *ARGS_PROTOCOLO]

    if args.prueba:
        tmp = tempfile.mkdtemp()
        for i, (m, _) in enumerate(REDES, 1):
            correr([sys.executable, ENTRENAR, "--datos", os.path.join(datos, ESTIRAR), "--modelo", m, "--fold", "1",
                    "--epocas", "1", "--limite", "512", "--salida", tmp, "--acumular", str(acum[m]), *comun],
                   f"prueba de {m} (partes de {args.lote // acum[m]})", i, len(REDES))
        print("\nPrueba lista: las tres redes caben; mirar la memoria GPU de cada una arriba", flush=True)
        return
    comun = ["--salida", salida, *comun]

    cv = trabajos_cv()
    pend = [t for t in cv if not os.path.exists(os.path.join(salida, nombre(*t) + ".json"))]
    finales = [m for m, _ in REDES if not os.path.exists(os.path.join(salida, nombre(m, 0, ESTIRAR) + ".json"))]
    n = len(pend) + len(finales)
    print(f"{len(cv)} corridas de validacion cruzada ({len(cv) - len(pend)} ya hechas) y {len(REDES)} finales en test "
          f"({len(REDES) - len(finales)} ya hechas); lote {args.lote}, en partes de "
          + ", ".join(f"{m} {args.lote // a}" for m, a in acum.items()), flush=True)
    t0 = time.time()
    for i, (m, f, c) in enumerate(pend, 1):
        correr([sys.executable, ENTRENAR, "--datos", os.path.join(datos, c), "--modelo", m, "--fold", str(f),
                "--epocas", str(EPOCAS_CV), "--acumular", str(acum[m]), *comun], nombre(m, f, c), i, n)
    for j, m in enumerate(finales, len(pend) + 1):
        e, ep = epocas_final(salida, m)
        print(f"\n{m}: mejores epocas en los folds {ep}; el entrenamiento final usa {e}", flush=True)
        correr([sys.executable, ENTRENAR, "--datos", os.path.join(datos, ESTIRAR), "--modelo", m, "--fold", "0",
                "--epocas", str(e), "--acumular", str(acum[m]), "--guardar-pesos", *comun], nombre(m, 0, ESTIRAR), j, n)
    print(f"\nTodo listo en {(time.time() - t0) / 3600:.1f} h\n", flush=True)
    subprocess.run([sys.executable, RESUMEN, salida])


if __name__ == "__main__":
    main()
