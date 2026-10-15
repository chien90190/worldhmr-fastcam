#!/usr/bin/env bash
# Installs the pipeline into a conda env. Run from the repository root:
#
#   bash scripts/install.sh                 creates the env worldhmr-fastcam and installs into it
#   bash scripts/install.sh --env NAME      another env name
#   bash scripts/install.sh --no-create     installs into an existing env
#
# Tested with the versions below. Each step is a function; to install something more (another
# HMR backbone or odometry that runs in this env), add a function and call it at the bottom.
set -e

ENV=worldhmr-fastcam
CREATE=true
while [ $# -gt 0 ]; do
  case $1 in
    --env) ENV=$2; shift 2 ;;
    --no-create) CREATE=false; shift ;;
    *) sed -n '2,9p' "$0"; exit 1 ;;
  esac
done

PYTHON=3.12.9
CUDA=cu126
TORCH="torch==2.6.0 torchvision==0.21.0"
TORCH_SCATTER_INDEX=https://data.pyg.org/whl/torch-2.6.0+$CUDA.html
XFORMERS=xformers==0.0.29.post2
SAM2_WHEEL=data/wheels/sam2-1.6-cp312-cp312-linux_x86_64.whl
SAM2_WHEEL_ID=1BaMz1U-mOIdr2gq04QKs7oGjEdNeqBen     # Google Drive file id, from PromptHMR

activate_env() {
  eval "$(conda shell.bash hook)"
  if [ "$CREATE" = true ]; then
    conda create -n "$ENV" python=$PYTHON -y
  fi
  conda activate "$ENV"
}

install_ffmpeg() {
  # ffmpeg and ffprobe, which read the input video
  conda install -y ffmpeg
}

install_torch() {
  pip install $TORCH --index-url https://download.pytorch.org/whl/$CUDA
  pip install --upgrade setuptools pip
  pip install torch-scatter -f $TORCH_SCATTER_INDEX
}

install_requirements() {
  pip install -r requirements.txt
  # pyrender pins PyOpenGL 3.1.0, which fails to create textures on Python 3.12; pip warns
  # about the pin, rendering works
  pip install PyOpenGL==3.1.7
}

install_chumpy() {
  # SMPL's pickle needs chumpy; this fork installs on numpy 2
  [ -d python_libs/chumpy ] || git clone https://github.com/Arthur151/chumpy python_libs/chumpy
  pip install -e python_libs/chumpy --no-build-isolation
}

install_xformers() {
  pip install -U $XFORMERS --index-url https://download.pytorch.org/whl/$CUDA --no-deps
}

install_sam2() {
  [ -f $SAM2_WHEEL ] || gdown -O $SAM2_WHEEL $SAM2_WHEEL_ID
  pip install $SAM2_WHEEL
}

mkdir -p python_libs data/wheels
activate_env
install_ffmpeg
install_torch
install_requirements
install_chumpy
install_xformers
install_sam2
echo "Installed into the conda env $ENV."
