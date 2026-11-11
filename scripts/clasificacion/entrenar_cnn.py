#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ajuste por transferencia de una CNN preentrenada en ImageNet (VGG16, ResNet50 o EfficientNet-B0 de
torchvision) para clasificar los recortes en VT, LP, TR y rechazo, con un fold de la validacion
cruzada de Jose. Guarda un JSON con la configuracion, las versiones y las metricas de cada epoca.

Uso (Colab): python entrenar_cnn.py --datos /content/clasificacion_colab --modelo vgg16 --fold 1
             --salida /content/drive/MyDrive/tesis/resultados_cnn [--tamano estirar|rellenar]
             [--pesos-clase] [--ajuste-completo] [--epocas 10] [--guardar-pesos]
"""

import argparse
import json
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASES = ["VT", "LP", "TR", "rechazo"]
VOLCANICAS = ["VT", "LP", "TR"]
# propuesta enviada al profesor: AV, IC y los recortes que no tocan ninguna caja van al rechazo;
# los que tocan una caja con IoU menor que 0,3 ("fuera") no se usan
MAPA = {"VT": "VT", "LP": "LP", "TR": "TR", "AV": "rechazo", "IC": "rechazo", "sin caja": "rechazo"}
# lado de entrada con que se preentrenaron las tres redes (crop_size 224 de sus transforms); no se usa el
# recorte central de esas transforms porque cortaria el inicio y el fin del evento
LADO = 224
# columnas del espectrograma de una ventana completa de 300 s (465 en las imagenes de Jose)
ANCHO_VENTANA = 465
CABEZAS = {"vgg16": ("classifier", 6), "resnet50": ("fc", None), "efficientnet_b0": ("classifier", 1)}


class Recortes(Dataset):
    """Imagen de cada recorte llevada a LADO x LADO: estirada, o puesta a la izquierda de una ventana de 300 s en negro."""
    def __init__(self, dir_img, archivos, y, tamano, mean, std):
        self.dir_img, self.archivos, self.y, self.tamano = dir_img, list(archivos), list(y), tamano
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.dir_img, self.archivos[i])).convert("RGB")
        if self.tamano == "rellenar" and im.width < ANCHO_VENTANA:
            fondo = Image.new("RGB", (ANCHO_VENTANA, im.height))
            fondo.paste(im, (0, 0))
            im = fondo
        im = im.resize((LADO, LADO), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        return (x - self.mean) / self.std, self.y[i]


def construir(modelo, n_clases, preentrenado=True):
    """Red de torchvision con sus pesos por defecto y la ultima capa cambiada por una de n_clases salidas."""
    pesos = torchvision.models.get_model_weights(modelo).DEFAULT if preentrenado else None
    red = torchvision.models.get_model(modelo, weights=pesos)
    attr, i = CABEZAS[modelo]
    vieja = getattr(red, attr) if i is None else getattr(red, attr)[i]
    nueva = nn.Linear(vieja.in_features, n_clases)
    if i is None:
        setattr(red, attr, nueva)
    else:
        getattr(red, attr)[i] = nueva
    t = pesos.transforms() if pesos is not None else None
    # media y desviacion con que se normalizaron las imagenes al preentrenar la red
    mean = list(t.mean) if t is not None else [0.485, 0.456, 0.406]
    std = list(t.std) if t is not None else [0.229, 0.224, 0.225]
    return red, nueva, mean, std, (pesos.name if pesos is not None else None)


def modo_entrenamiento(red, cabeza, congelada):
    """Con la red congelada solo la capa nueva entrena; el resto queda en modo evaluacion (BatchNorm y Dropout fijos)."""
    red.train()
    if congelada:
        for m in red.modules():
            m.eval()
        cabeza.train()


def metricas(c):
    """Desde la matriz de confusion (filas: clase real; columnas: prediccion)."""
    out = {}
    for i, k in enumerate(CLASES):
        tp, real, pred = c[i, i], c[i].sum(), c[:, i].sum()
        rec = tp / real if real else float("nan")
        pre = tp / pred if pred else float("nan")
        f1 = 2 * pre * rec / (pre + rec) if pred and real and (pre + rec) else 0.0
        out[k] = dict(n=int(real), sensibilidad=float(rec), precision=float(pre), f1=float(f1))
    out["f1_macro_volcanicas"] = float(np.mean([out[k]["f1"] for k in VOLCANICAS]))
    out["exactitud"] = float(np.trace(c) / c.sum())
    return out


def evaluar(red, carga, disp):
    red.eval()
    c = np.zeros((len(CLASES), len(CLASES)), dtype=np.int64)
    with torch.no_grad():
        for x, y in carga:
            p = red(x.to(disp)).argmax(1).cpu().numpy()
            np.add.at(c, (y.numpy(), p), 1)
    return c


def semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, help="carpeta con indice.csv e imagenes/")
    ap.add_argument("--modelo", required=True, choices=sorted(CABEZAS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--tamano", default="estirar", choices=["estirar", "rellenar"])
    ap.add_argument("--pesos-clase", action="store_true", help="pesa la perdida por el inverso de la frecuencia de cada clase")
    ap.add_argument("--ajuste-completo", action="store_true", help="entrena toda la red, no solo la capa nueva")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--trabajadores", type=int, default=2)
    ap.add_argument("--limite", type=int, default=None, help="solo N recortes por particion, para probar")
    ap.add_argument("--sin-preentrenar", action="store_true", help="pesos al azar, solo para probar sin descargar")
    ap.add_argument("--guardar-pesos", action="store_true")
    args = ap.parse_args()
    semilla(args.semilla)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = pd.read_csv(os.path.join(args.datos, "indice.csv"))
    idx = idx[idx.etiqueta.isin(MAPA)].copy()
    idx["y"] = idx.etiqueta.map(MAPA).map({k: i for i, k in enumerate(CLASES)})
    ent = idx[(idx.split == "desarrollo") & (idx.fold != args.fold)]
    val = idx[idx.fold == args.fold]
    if args.limite:
        ent, val = ent.sample(min(args.limite, len(ent)), random_state=0), val.sample(min(args.limite, len(val)), random_state=0)
    if set(ent.ventana) & set(val.ventana):
        raise SystemExit("DETENIDO: hay ventanas en entrenamiento y validacion a la vez")

    red, cabeza, mean, std, nombre_pesos = construir(args.modelo, len(CLASES), not args.sin_preentrenar)
    congelada = not args.ajuste_completo
    if congelada:
        for p in red.parameters():
            p.requires_grad = False
        for p in cabeza.parameters():
            p.requires_grad = True
    red.to(disp)
    dir_img = os.path.join(args.datos, "imagenes")
    kw = dict(batch_size=args.lote, num_workers=args.trabajadores, pin_memory=disp.type == "cuda")
    gen = torch.Generator().manual_seed(args.semilla)
    c_ent = DataLoader(Recortes(dir_img, ent.archivo, ent.y, args.tamano, mean, std), shuffle=True, generator=gen, **kw)
    c_val = DataLoader(Recortes(dir_img, val.archivo, val.y, args.tamano, mean, std), shuffle=False, **kw)

    n_por_clase = np.bincount(ent.y, minlength=len(CLASES))
    peso = torch.tensor(n_por_clase.sum() / (len(CLASES) * n_por_clase), dtype=torch.float32) if args.pesos_clase else None
    perdida = nn.CrossEntropyLoss(weight=peso.to(disp) if peso is not None else None)
    # AdamW con sus valores por defecto de PyTorch (lr 0,001, weight_decay 0,01)
    opt = torch.optim.AdamW([p for p in red.parameters() if p.requires_grad])

    nombre = f"{args.modelo}_fold{args.fold}_{args.tamano}" + ("_pesos" if args.pesos_clase else "") + \
             ("_completo" if args.ajuste_completo else "")
    cfg = dict(vars(args), clases=CLASES, mapa=MAPA, lado=LADO, pesos_preentrenados=nombre_pesos, mean=mean, std=std,
               congelada=congelada, n_entrenamiento={k: int(n) for k, n in zip(CLASES, n_por_clase)},
               n_validacion={k: int(n) for k, n in zip(CLASES, np.bincount(val.y, minlength=len(CLASES)))},
               versiones=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__),
               dispositivo=torch.cuda.get_device_name(0) if disp.type == "cuda" else "cpu")
    print(f"{nombre}: entrenamiento {len(ent)}, validacion {len(val)} (ventanas {ent.ventana.nunique()} y "
          f"{val.ventana.nunique()}); {cfg['dispositivo']}; pesos {nombre_pesos}", flush=True)
    print(f"  por clase en entrenamiento: {cfg['n_entrenamiento']}", flush=True)

    historia, mejor = [], None
    for ep in range(1, args.epocas + 1):
        t0 = time.time()
        modo_entrenamiento(red, cabeza, congelada)
        suma, n = 0.0, 0
        for x, y in c_ent:
            x, y = x.to(disp), y.to(disp)
            opt.zero_grad()
            l = perdida(red(x), y)
            l.backward()
            opt.step()
            suma, n = suma + l.item() * len(y), n + len(y)
        c = evaluar(red, c_val, disp)
        m = metricas(c)
        historia.append(dict(epoca=ep, perdida=suma / n, segundos=time.time() - t0, matriz=c.tolist(), **m))
        print(f"  epoca {ep}: perdida {suma / n:.3f}; validacion: F1 macro volcanicas {m['f1_macro_volcanicas']:.3f}; "
              + ", ".join(f"{k} sens {m[k]['sensibilidad']:.3f}" for k in CLASES) + f"; {time.time() - t0:.0f} s", flush=True)
        if mejor is None or m["f1_macro_volcanicas"] > mejor["f1_macro_volcanicas"]:
            mejor = historia[-1]
            if args.guardar_pesos:
                os.makedirs(args.salida, exist_ok=True)
                torch.save(red.state_dict(), os.path.join(args.salida, f"{nombre}.pt"))

    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, f"{nombre}.json"), "w") as f:
        json.dump(dict(configuracion=cfg, historia=historia, mejor_epoca=mejor["epoca"]), f, indent=1)
    print(f"Mejor epoca {mejor['epoca']} (F1 macro volcanicas {mejor['f1_macro_volcanicas']:.3f}); matriz de confusion "
          f"(filas real, columnas prediccion; {', '.join(CLASES)}):")
    for k, fila in zip(CLASES, mejor["matriz"]):
        print(f"  {k:>8} {fila}")
    print(f"Resultados en {os.path.join(args.salida, nombre + '.json')}")


if __name__ == "__main__":
    main()
