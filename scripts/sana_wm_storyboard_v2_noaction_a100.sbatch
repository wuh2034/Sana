#!/bin/bash
# storyboard_v2 prompt, NO action (static camera 'none-320'). Only the prompt drives the video.
# A100 80GB partition (free cards available); offload flags added for VRAM safety (output-neutral).
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=sana-wm-v2-noaction
#SBATCH -o /dss/dsshome1/0F/go29sal2/thesis/logs/sana_wm_v2_noaction_%j.out
#SBATCH -e /dss/dsshome1/0F/go29sal2/thesis/logs/sana_wm_v2_noaction_%j.err

set -eo pipefail

DSS=/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go29sal2
HOME_ROOT=/dss/dsshome1/0F/go29sal2/thesis
SANA_ROOT=$HOME_ROOT/Sana

set +u
source "$DSS/miniforge3/etc/profile.d/conda.sh"
conda activate sana
set -u

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=$DSS/hf-cache
export TOKENIZERS_PARALLELISM=false
export DISABLE_XFORMERS=1

cd "$SANA_ROOT"

echo "==================== env ===================="
date; hostname
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('total_GB', round(torch.cuda.get_device_properties(0).total_memory/1e9,2))"

echo "==================== prompt ===================="
cat asset/sana_wm/car_storyboard_v2.txt

echo "==================== run (NO action -> static camera 'none-320') ===================="
time python inference_video_scripts/inference_sana_wm.py \
  --image asset/sana_wm/car_ball.png \
  --prompt asset/sana_wm/car_storyboard_v2.txt \
  --action "none-320" \
  --num_frames 321 \
  --output_dir results/sana_wm_demo_car_5_storyboard_v2_noaction \
  --name storyboard_v2_noaction \
  --offload_vae \
  --offload_refiner

echo "==================== done ===================="
date
