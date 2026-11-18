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

    import duraciones_recortes as dr
    dur = pd.Series([1.0, 3.0, 6.0, 12.0, 50.0, 70.0])
    et = pd.Series(["VT", "VT", "VT", "TR", "TR", "TR"])
    fm = dr.fraccion(dur, et, [5], mayor=False)
    ff = dr.fraccion(dur, et, [60], mayor=True)
    todo &= caso("duraciones: parte mas corta que el minimo y parte que se cortaria, por etiqueta",
                 abs(fm.loc["VT", "5 s"] - 2 / 3) < 1e-3 and fm.loc["TR", "5 s"] == 0
                 and ff.loc["VT", "60 s"] == 0 and abs(ff.loc["TR", "60 s"] - 1 / 3) < 1e-3
                 and dr.percentiles(dur, et).loc["TR", "n"] == 3)

    import espectrogramas_fijos as ef
    todo &= caso("tramo fijo: desde el inicio del recorte; al final de la ventana, sus ultimos segundos",
                 ef.tramo_fijo(10.0, 60.0) == (10.0, 70.0) and ef.tramo_fijo(280.0, 60.0) == (240.0, 300.0)
                 and ef.tramo_fijo(0.0, 60.0) == (0.0, 60.0))
    todo &= caso("carpeta dentro del zip: el nombre del zip",
                 pc.raiz("/x/clasificacion_colab_fijo60s.zip") == "clasificacion_colab_fijo60s")

    import resumen_cnn as rs
    import correr_todo as ct
    d = tempfile.mkdtemp()
    for f, ep in zip(ct.FOLDS, [10, 7, 9, 7, 8]):
        json.dump(dict(mejor_epoca=ep), open(os.path.join(d, ct.nombre("resnet50", f, ct.ESTIRAR) + ".json"), "w"))
    todo &= caso("corredor: 20 corridas de folds y epocas finales = mediana de las mejores (10, 7, 9, 7, 8 -> 8)",
                 len(ct.trabajos_cv()) == 20 and ct.epocas_final(d, "resnet50") == (8, [10, 7, 9, 7, 8])
                 and ct.nombre("efficientnet_b0", 0, ct.FIJO) == "efficientnet_b0_test_estirar_fijo60s_pesos_completo_lr0.0001_min5s")
    m = {"f1_macro_volcanicas": 0.7, "matriz": [[8, 1, 0, 1], [0, 9, 0, 1], [0, 0, 4, 1], [2, 1, 0, 7]]}
    json.dump(dict(configuracion=dict(modelo="vgg16", fold=0, conjunto="recorte", evaluacion="test",
                                      cortos_fuera_de_validacion={"VT": 0, "LP": 0, "TR": 0, "rechazo": 10}),
                   historia=[dict(m, segundos=5.0)], mejor_epoca=1), open(os.path.join(d, "x.json"), "w"))
    r = rs.leer(os.path.join(d, "x.json"))
    todo &= caso("resumen: los recortes cortos cuentan como rechazados (3 falsos de 20) y la sensibilidad sobre el total",
                 abs(r["falsos"] - 3 / 20) < 1e-12 and abs(r["volcanicos"] - 21 / 25) < 1e-12 and r["sens_VT"] == 0.8
                 and r["n_rechazo"] == 20)

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
                          fold=(i // 3) % 5 + 1 if i < 48 else pd.NA, dur_s=10.0 + i if i < 48 else 1.0 + 2 * (i - 48)))
    pd.DataFrame(filas).to_csv(os.path.join(datos, "indice.csv"), index=False)
    # sin 'fuera' quedan 40 de desarrollo: 7 del fold 2 (validacion) y 33 de los otros folds
    for modelo, extra, archivo, congelada in (
            ("efficientnet_b0", ["--pesos-clase"], "efficientnet_b0_fold2_estirar_pesos.json", True),
            ("resnet50", ["--tamano", "rellenar", "--ajuste-completo"], "resnet50_fold2_rellenar_completo.json", False),
            ("efficientnet_b0", ["--ajuste-completo", "--lr", "0.0001"], "efficientnet_b0_fold2_estirar_completo_lr0.0001.json", False),
            ("vgg16", ["--ajuste-completo", "--acumular", "4"], "vgg16_fold2_estirar_completo.json", False)):
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
                  and sum(map(sum, res["historia"][0]["matriz"])) == 7
                  and cfg["lr_usada"] == (1e-4 if "--lr" in extra else 1e-3))
        else:
            print(r.stderr[-800:])
        todo &= caso(f"corrida corta {modelo} {' '.join(extra)}: fold 2 a validacion, 'fuera' y test fuera, JSON con 2 epocas", ok)

    # el mismo conjunto con nombre de paquete de duracion fija y una duracion minima de 30 s:
    # de la validacion del fold 2 salen 4 recortes de 13, 14, 28 y 29 s (VT, LP y dos de rechazo)
    fijo = os.path.join(tempfile.mkdtemp(), "clasificacion_colab_fijo60s")
    os.symlink(datos, fijo)
    salida = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", fijo, "--modelo", "efficientnet_b0",
                        "--fold", "2", "--salida", salida, "--epocas", "1", "--lote", "8", "--trabajadores", "0",
                        "--sin-preentrenar", "--duracion-minima", "30"], capture_output=True, text=True)
    ruta = os.path.join(salida, "efficientnet_b0_fold2_estirar_fijo60s_min30s.json")
    ok = r.returncode == 0 and os.path.exists(ruta)
    if ok:
        cfg = json.load(open(ruta))["configuracion"]
        ok = (cfg["conjunto"] == "fijo60s" and sum(cfg["n_validacion"].values()) == 3
              and cfg["cortos_fuera_de_validacion"] == {"VT": 1, "LP": 1, "TR": 0, "rechazo": 2})
    else:
        print(r.stderr[-800:])
    todo &= caso("conjunto de duracion fija en el nombre y duracion minima que saca los recortes cortos", ok)

    from torch import nn
    x = torch.randn(8, 5, generator=torch.Generator().manual_seed(0))
    y = torch.tensor([0, 1, 2, 3, 0, 0, 3, 3])
    peso = torch.tensor([2.0, 1.0, 0.5, 0.25])

    def un_paso(k):
        torch.manual_seed(1)
        red = nn.Linear(5, 4)
        ref = nn.CrossEntropyLoss(weight=peso)(red(x), y).item()
        opt = torch.optim.AdamW(red.parameters(), lr=0.1)
        t = 8 // k
        l = ec.paso(red, nn.CrossEntropyLoss(weight=peso, reduction="sum"), peso,
                    [(x[i:i + t], y[i:i + t]) for i in range(0, 8, t)], opt, torch.device("cpu"))
        return red.weight.detach().clone(), l, ref

    w1, l1, ref = un_paso(1)
    w2, l2, _ = un_paso(2)
    w4, l4, _ = un_paso(4)
    todo &= caso("lote en 1, 2 o 4 partes: misma perdida ponderada y mismos pesos despues del paso",
                 abs(l1 - ref) < 1e-6 and abs(l2 - ref) < 1e-6 and abs(l4 - ref) < 1e-6
                 and torch.allclose(w1, w2, atol=1e-6) and torch.allclose(w1, w4, atol=1e-6))

    todo &= caso("el corredor nombra los resultados igual que entrenar_cnn.py",
                 all(ct.nombre(m, f, c) == ec.nombre_resultado(m, f, "estirar", ec.conjunto_de(c), True, True, 1e-4, 5.0)
                     for m, f, c in ct.trabajos_cv() + [(m, 0, ct.ESTIRAR) for m, _ in ct.REDES]))

    # fold 0: entrena con todo desarrollo (40 sin 'fuera') y evalua en test (10); con minimo 5 s salen de test
    # los de 1 y 3 s (un VT y un LP); informa la ultima epoca y guarda los pesos
    salida = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", datos, "--modelo", "efficientnet_b0",
                        "--fold", "0", "--salida", salida, "--epocas", "2", "--lote", "8", "--trabajadores", "0",
                        "--sin-preentrenar", "--duracion-minima", "5", "--guardar-pesos"], capture_output=True, text=True)
    ruta = os.path.join(salida, "efficientnet_b0_test_estirar_min5s.json")
    ok = r.returncode == 0 and os.path.exists(ruta) and os.path.exists(ruta[:-5] + ".pt")
    if ok:
        res = json.load(open(ruta))
        cfg = res["configuracion"]
        ok = (cfg["evaluacion"] == "test" and res["mejor_epoca"] == 2 and sum(cfg["n_validacion"].values()) == 8
              and sum(cfg["n_entrenamiento"].values()) == 40
              and cfg["cortos_fuera_de_validacion"] == {"VT": 1, "LP": 1, "TR": 0, "rechazo": 0})
    else:
        print(r.stderr[-800:])
    todo &= caso("fold 0: entrena con desarrollo, evalua en test la ultima epoca, cuenta los cortos de test y guarda pesos", ok)

    print("TODOS PASAN" if todo else "HAY FALLAS")
    sys.exit(0 if todo else 1)


if __name__ == "__main__":
    main()
