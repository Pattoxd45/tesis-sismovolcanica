#!/bin/bash
#SBATCH --job-name=busqueda
#SBATCH --partition=RunCuda
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
# Busqueda de hiperparametros de una red en el fold 1. Uso: sbatch busqueda.sbatch efficientnet_b0
# Si se corta, se vuelve a lanzar igual y sigue donde quedo (las pruebas quedan en un archivo SQLite).
# sin --gres: el nodo no tiene gres configurado y sbatch lo rechaza (pendiente con el equipo del cluster)
set -euo pipefail
MODELO="$1"
# limites del aumento de datos (decision propia, por revisar con el profesor)
AUMENTO="--jitter-maximo 5 --jitter-ancho 50 70 --elastica 50 0"
source ~/cnn/.venv/bin/activate
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python ~/cnn/scripts/clasificacion/buscar_cnn.py --datos ~/cnn/clasificacion_db --modelo "$MODELO" \
    --salida ~/cnn/resultados/busqueda --pruebas 20 --trabajadores 3 $AUMENTO
