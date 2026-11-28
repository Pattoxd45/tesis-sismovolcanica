#!/bin/bash
#SBATCH --job-name=folds
#SBATCH --partition=RunCuda
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
# 5 folds con la mejor combinacion de la busqueda de una red, con el recorte y con la ventana fija de 60 s,
# y el mejor modelo de cada fold evaluado en test. Uso: sbatch folds.sbatch efficientnet_b0
# Si se corta, se vuelve a lanzar igual: las corridas que ya tienen su JSON no se repiten.
# sin --gres: el nodo no tiene gres configurado y sbatch lo rechaza (pendiente con el equipo del cluster)
set -euo pipefail
MODELO="$1"
# los mismos limites del aumento que en la busqueda
AUMENTO="--jitter-maximo 5 --jitter-ancho 50 70 --elastica 50 0"
source ~/cnn/.venv/bin/activate
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python ~/cnn/scripts/clasificacion/folds_cnn.py --datos ~/cnn/clasificacion_db --modelo "$MODELO" \
    --busqueda ~/cnn/resultados/busqueda --salida ~/cnn/resultados/folds --trabajadores 3 $AUMENTO
