#!/bin/bash
#SBATCH --job-name=prueba_gpu
#SBATCH --partition=RunCuda
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
# Prueba corta: PyTorch ve la GPU y multiplica dos matrices en ella. Uso: sbatch prueba_gpu.sbatch
# sin --gres: el nodo no tiene gres configurado y sbatch lo rechaza (pendiente con el equipo del cluster)
set -euo pipefail
source ~/cnn/.venv/bin/activate
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
python -c "
import torch, torchvision
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.version.cuda)
print('gpu disponible', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device='cuda')
print('suma', (x @ x).sum().item(), 'memoria GB', round(torch.cuda.max_memory_allocated() / 1e9, 2))
"
