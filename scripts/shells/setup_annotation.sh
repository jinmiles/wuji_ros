#!/usr/bin/env bash
# New PC (robot room PC etc.): the analysis env and assets that ./annotate.sh needs (ANNOTATION.md "설치").
# Run once from a fresh clone, after putting MANO_RIGHT.pkl in assets/mano/ by hand (USB etc.; the new
# PC has no SSH to the analysis server). A re-run skips the parts already done.
# Usage: setup_annotation.sh
set -eo pipefail
(($# == 0)) || { echo "usage: setup_annotation.sh  (no arguments)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
# Same as annotate.sh: the env's packages only, not the recording PC's user site or a sourced ROS 2 PYTHONPATH.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

for cmd in conda git wget nvidia-smi; do
  command -v "$cmd" >/dev/null || { echo "$cmd not found (ANNOTATION.md: 필요한 것)" >&2; exit 1; }
done
command -v ffmpeg >/dev/null || echo "warning: no ffmpeg, annotate will write jpg only (sudo apt install ffmpeg)"
nvidia-smi | grep -o "CUDA Version: [0-9.]*"  # the env's torch is a CUDA 11.8 build

# MANO first, so a missing pkl stops before the long steps (conda env, 6 GB download).
echo "== 1. MANO"
mano=assets/mano/MANO_RIGHT.pkl
[[ -f "$mano" ]] || { echo "no $mano: copy MANO_RIGHT.pkl there first (ANNOTATION.md 설치)" >&2; exit 1; }
echo "fd5a9d35f914987cf1cc04ffe338caa1  $mano" | md5sum -c -
mkdir -p assets/hamer/data
ln -sfn ../../mano assets/hamer/data/mano  # HaMeR reads the same pkl

echo "== 2. HaMeR code"
git submodule update --init third_party/hamer

echo "== 3. conda env wuji_ros"
if ! conda env list | awk '{print $1}' | grep -x wuji_ros >/dev/null; then
  conda env create -f environment.yml
fi
if ! conda run -n wuji_ros python -m pip show -q chumpy 2>/dev/null; then
  conda run --no-capture-output -n wuji_ros python -m pip install --no-build-isolation \
      "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
fi

echo "== 4. HaMeR checkpoint"
ckpt=assets/hamer/hamer_ckpts/checkpoints/hamer.ckpt
mean=assets/hamer/data/mano_mean_params.npz
if [[ ! -f "$ckpt" || ! -f "$mean" ]]; then
  mkdir -p scratch
  wget -c -P scratch https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
  tar --warning=no-unknown-keyword --exclude=".*" -xzf scratch/hamer_demo_data.tar.gz -C scratch \
      _DATA/hamer_ckpts _DATA/data/mano_mean_params.npz
  rm -rf assets/hamer/hamer_ckpts  # only a partial copy from an interrupted run can be here
  mv scratch/_DATA/hamer_ckpts assets/hamer/hamer_ckpts
  mv scratch/_DATA/data/mano_mean_params.npz "$mean"
  rm -r scratch/_DATA scratch/hamer_demo_data.tar.gz
fi

echo "== 5. check"
conda run --no-capture-output -n wuji_ros python -c "
import torch, smplx, chumpy, pytorch_lightning, cv2, trimesh, numpy
got = (torch.__version__, torch.cuda.is_available(), numpy.__version__)
print(*got)
assert got == ('2.0.1+cu118', True, '1.23.5'), 'expected 2.0.1+cu118 True 1.23.5'"
ls -lL "$ckpt" assets/hamer/data/mano/MANO_RIGHT.pkl
echo "annotation env ok"
