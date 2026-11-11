#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para paquete_colab.py y entrenar_cnn.py: folds de Jose por ventana, metricas desde la
matriz de confusion, imagen de entrada y una corrida corta con pesos al azar (sin GPU ni descargas).

Uso: python test_clasificacion.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import paquete_colab as pc


def caso(nombre, ok):
    print(f"  {'OK   ' if ok else 'FALLA'} {nombre}")
    return ok


def splits(d, val, desarrollo):
    for k, vs in enumerate(val, 1):
        with open(os.path.join(d, f"fold_{k:02d}_val.txt"), "w") as f:
            f.writelines(f"/ruta/{v}_300s.png\n" for v in vs)
    with open(os.path.join(d, "desarrollo.txt"), "w") as f:
        f.writelines(f"/ruta/{v}_300s.png\n" for v in desarrollo)


def detenido(f, *a):
    try:
        f(*a)
        return False
    except SystemExit:
        return True


def main():
    todo = True
    d = tempfile.mkdtemp()
    splits(d, [["a"], ["b"], ["c"], ["d"], ["e", "f"]], list("abcdef"))
    fold = pc.folds_jose(d)
    todo &= caso("fold de validacion de cada ventana de desarrollo", fold == dict(a=1, b=2, c=3, d=4, e=5, f=5))
    splits(d, [["a"], ["a"], ["c"], ["d"], ["e", "f"]], list("acdef"))
    todo &= caso("ventana en dos folds: se detiene", detenido(pc.folds_jose, d))
    splits(d, [["a"], ["b"], ["c"], ["d"], ["e"]], list("abcdef"))
    todo &= caso("folds que no cubren desarrollo: se detiene", detenido(pc.folds_jose, d))
    idx = pd.DataFrame(dict(ventana=["a", "a", "t"], split=["desarrollo", "desarrollo", "test"]))
    todo &= caso("recortes de una ventana con su fold; test sin fold",
                 pc.indice_con_folds(idx, dict(a=3)).fold.astype("object").fillna(0).tolist() == [3, 3, 0])
    todo &= caso("ventana de desarrollo sin fold: se detiene",
                 detenido(pc.indice_con_folds, pd.DataFrame(dict(ventana=["z"], split=["desarrollo"])), dict(a=3)))

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        print("  (sin PyTorch o torchvision: se omiten los casos de entrenar_cnn.py)")
        print("TODOS PASAN" if todo else "HAY FALLAS")
        sys.exit(0 if todo else 1)
    import entrenar_cnn as ec

    c = np.array([[8, 1, 1, 0], [0, 5, 0, 5], [0, 0, 0, 2], [2, 0, 0, 16]])
    m = ec.metricas(c)
    todo &= caso("metricas: sensibilidad, precision y F1 por clase; F1 macro solo de VT, LP y TR",
                 abs(m["VT"]["sensibilidad"] - 0.8) < 1e-12 and abs(m["VT"]["precision"] - 0.8) < 1e-12
                 and abs(m["LP"]["f1"] - 2 * (5 / 6) * 0.5 / (5 / 6 + 0.5)) < 1e-12 and m["TR"]["f1"] == 0.0
                 and abs(m["f1_macro_volcanicas"] - (0.8 + m["LP"]["f1"]) / 3) < 1e-12 and abs(m["exactitud"] - 29 / 40) < 1e-12)

    di = tempfile.mkdtemp()
    Image.fromarray(np.full((231, 40, 3), 200, dtype=np.uint8)).save(os.path.join(di, "r.png"))
    for tam in ("estirar", "rellenar"):
        x, y = ec.Recortes(di, ["r.png"], [2], tam, [0, 0, 0], [1, 1, 1])[0]
        negro = float((x.sum(0) == 0).float().mean())
        todo &= caso(f"imagen {tam}: 3 x 224 x 224; " + ("sin fondo negro" if tam == "estirar" else "el recorte ocupa 40 de 465 columnas"),
                     tuple(x.shape) == (3, 224, 224) and y == 2
                     and (negro == 0 if tam == "estirar" else abs((1 - negro) - 40 / 465) < 0.02))

    for mod in ec.CABEZAS:
        red, cabeza, *_ = ec.construir(mod, 4, preentrenado=False)
        todo &= caso(f"{mod}: la capa nueva tiene 4 salidas", red(torch.zeros(1, 3, 224, 224)).shape == (1, 4)
                     and cabeza.out_features == 4)

    red, cabeza, *_ = ec.construir("resnet50", 4, preentrenado=False)
    ec.modo_entrenamiento(red, cabeza, congelada=True)
    todo &= caso("red congelada: BatchNorm en evaluacion y la capa nueva entrenando",
                 not red.bn1.training and cabeza.training)

    datos = tempfile.mkdtemp()
    os.makedirs(os.path.join(datos, "imagenes"))
    rng = np.random.default_rng(0)
    filas = []
    for i in range(60):
        et = ["VT", "LP", "TR", "IC", "sin caja", "fuera"][i % 6]
        a = f"v{i // 3}_{i % 3:02d}.png"
        Image.fromarray(rng.integers(0, 255, (231, 10 + i, 3), dtype=np.uint8)).save(os.path.join(datos, "imagenes", a))
        filas.append(dict(archivo=a, ventana=f"v{i // 3}", etiqueta=et, split="desarrollo" if i < 48 else "test",
                          fold=(i // 3) % 5 + 1 if i < 48 else pd.NA))
    pd.DataFrame(filas).to_csv(os.path.join(datos, "indice.csv"), index=False)
    # sin 'fuera' quedan 40 de desarrollo: 7 del fold 2 (validacion) y 33 de los otros folds
    for modelo, extra, archivo, congelada in (
            ("efficientnet_b0", ["--pesos-clase"], "efficientnet_b0_fold2_estirar_pesos.json", True),
            ("resnet50", ["--tamano", "rellenar", "--ajuste-completo"], "resnet50_fold2_rellenar_completo.json", False)):
        salida = tempfile.mkdtemp()
        r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", datos, "--modelo", modelo,
                            "--fold", "2", "--salida", salida, "--epocas", "2", "--lote", "8", "--trabajadores", "0",
                            "--sin-preentrenar", *extra], capture_output=True, text=True)
        ok = r.returncode == 0
        if ok:
            res = json.load(open(os.path.join(salida, archivo)))
            cfg = res["configuracion"]
            ok = (len(res["historia"]) == 2 and sum(cfg["n_validacion"].values()) == 7
                  and sum(cfg["n_entrenamiento"].values()) == 33 and cfg["congelada"] == congelada
                  and sum(map(sum, res["historia"][0]["matriz"])) == 7)
        else:
            print(r.stderr[-800:])
        todo &= caso(f"corrida corta {modelo} {' '.join(extra)}: fold 2 a validacion, 'fuera' y test fuera, JSON con 2 epocas", ok)

    print("TODOS PASAN" if todo else "HAY FALLAS")
    sys.exit(0 if todo else 1)


if __name__ == "__main__":
    main()
