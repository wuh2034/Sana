#!/bin/bash
# SANA-WM single-GPU validation job.
# Defaults match the last successful A100 MIG 3g.40gb text+WASD retest.
#
# Submit:
#   sbatch /dss/dsshome1/0F/go29sal2/thesis/Sana/scripts/sana_wm_a100_mig_text_wasd.sbatch

# Same GPU request as the last run: one A100 MIG slice with 40 GB HBM.
# For a full 80 GB A100, edit the next lines to a full-A100 partition and use:
#   #SBATCH --gres=gpu:1
# instead of:
#   #SBATCH --gres=gpu:3g.40gb:1
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=sana-wm-a100-mig
#SBATCH -o /dss/dsshome1/0F/go29sal2/thesis/logs/sana_wm_a100_mig_%j.out
#SBATCH -e /dss/dsshome1/0F/go29sal2/thesis/logs/sana_wm_a100_mig_%j.err

set -eo pipefail

DSS=/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go29sal2
HOME_ROOT=/dss/dsshome1/0F/go29sal2/thesis
SANA_ROOT=$HOME_ROOT/Sana

# Edit these values to change the inference job.
# The generation knobs match the last run. OUTPUT_DIR includes the job id
# by default so repeated submissions do not overwrite older videos.
IMAGE=${IMAGE:-asset/sana_wm/demo_0.png}
PROMPT_FILE=${PROMPT_FILE:-$HOME_ROOT/outputs/sana_wm_original_path_retest_matched_prompt.txt}
INTRINSICS=${INTRINSICS:-asset/sana_wm/demo_0_intrinsics.npy}
ACTION=${ACTION:-w-50,d-15,lw-15}
NUM_FRAMES=${NUM_FRAMES:-81}
FPS=${FPS:-8}
STEP=${STEP:-12}
CFG_SCALE=${CFG_SCALE:-5.0}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME_ROOT/outputs/sana_wm_a100_mig_${SLURM_JOB_ID:-manual}}
RUN_NAME=${RUN_NAME:-original_path_matched_prompt}

# Some conda activation hooks read optional variables before defining them.
# Enable nounset only after the environment is active.
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

mkdir -p "$HOME_ROOT/logs" "$OUTPUT_DIR"
cd "$SANA_ROOT"

echo "==================== env ===================="
date
hostname
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('total_GB', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))"
echo "HF_HOME=$HF_HOME"

echo "==================== prompt ===================="
cat "$PROMPT_FILE"

echo
echo "==================== command ===================="
echo python inference_video_scripts/inference_sana_wm.py \
  --image "$IMAGE" \
  --prompt "$PROMPT_FILE" \
  --intrinsics "$INTRINSICS" \
  --action "$ACTION" \
  --num_frames "$NUM_FRAMES" \
  --fps "$FPS" \
  --step "$STEP" \
  --cfg_scale "$CFG_SCALE" \
  --output_dir "$OUTPUT_DIR" \
  --name "$RUN_NAME"

  # --no_refiner \
  # --offload_vae \

python inference_video_scripts/inference_sana_wm.py \
  --image "$IMAGE" \
  --prompt "$PROMPT_FILE" \
  --intrinsics "$INTRINSICS" \
  --action "$ACTION" \
  --num_frames "$NUM_FRAMES" \
  --fps "$FPS" \
  --step "$STEP" \
  --cfg_scale "$CFG_SCALE" \
  --output_dir "$OUTPUT_DIR" \
  --name "$RUN_NAME" \
  --no_refiner 
  # --offload_refiner 
  # --offload_vae \

echo "==================== result ===================="
ls -lh "$OUTPUT_DIR"
echo "[sana_wm_a100_mig_text_wasd] DONE"
