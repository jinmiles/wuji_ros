#!/usr/bin/env bash
# New PC (robot room PC etc.): the analysis env and assets that ./annotate.sh needs (ANNOTATION.md "설치").
# Run once from a fresh clone; a re-run skips the parts already done.
# Usage:
#   setup_annotation.sh [--from SRC]
#     SRC: scp source of a wuji_ros checkout that has assets/mano/MANO_RIGHT.pkl
#          (user@host:/path/to/wuji_ros, or a local path). Not needed when the pkl is already in place.
set -eo pipefail
usage="usage: setup_annotation.sh [--from user@host:/path/to/wuji_ros]"
from=""
while (($#)); do
  case "$1" in
    --from) from="${2:?$usage}"; shift 2 ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
done
cd "$(dirname "$0")/../.."
# Same as annotate.sh: the env's packages only, not the recording PC's user site or a sourced ROS 2 PYTHONPATH.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

for cmd in conda git wget nvidia-smi; do
  command -v "$cmd" >/dev/null || { echo "$cmd not found (ANNOTATION.md: 필요한 것)" >&2; exit 1; }
done
command -v ffmpeg >/dev/null || echo "warning: no ffmpeg, annotate will write jpg only (sudo apt install ffmpeg)"
nvidia-smi | grep -o "CUDA Version: [0-9.]*"  # the env's torch is a CUDA 11.8 build

echo "== 1. HaMeR code"
git submodule update --init third_party/hamer

echo "== 2. conda env wuji_ros"
if ! conda env list | awk '{print $1}' | grep -x wuji_ros >/dev/null; then
  conda env create -f environment.yml
fi
if ! conda run -n wuji_ros python -m pip show -q chumpy 2>/dev/null; then
  conda run --no-capture-output -n wuji_ros python -m pip install --no-build-isolation \
      "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
fi

echo "== 3. HaMeR checkpoint"
ckpt=assets/hamer/hamer_ckpts/checkpoints/hamer.ckpt
mean=assets/hamer/data/mano_mean_params.npz
if [[ ! -f "$ckpt" || ! -f "$mean" ]]; then
  mkdir -p scratch assets/hamer/data
  wget -c -P scratch https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
  tar --warning=no-unknown-keyword --exclude=".*" -xzf scratch/hamer_demo_data.tar.gz -C scratch \
      _DATA/hamer_ckpts _DATA/data/mano_mean_params.npz
  rm -rf assets/hamer/hamer_ckpts  # only a partial copy from an interrupted run can be here
  mv scratch/_DATA/hamer_ckpts assets/hamer/hamer_ckpts
  mv scratch/_DATA/data/mano_mean_params.npz "$mean"
  rm -r scratch/_DATA scratch/hamer_demo_data.tar.gz
fi

echo "== 4. MANO"
mano=assets/mano/MANO_RIGHT.pkl
if [[ ! -f "$mano" ]]; then
  [[ -n "$from" ]] || { echo "no $mano: rerun with --from, or put the MANO site's MANO_RIGHT.pkl there" >&2; exit 1; }
  mkdir -p assets/mano
  scp "$from/$mano" "$mano"
fi
echo "fd5a9d35f914987cf1cc04ffe338caa1  $mano" | md5sum -c -
ln -sfn ../../mano assets/hamer/data/mano

echo "== 5. check"
conda run --no-capture-output -n wuji_ros python -c "
import torch, smplx, chumpy, pytorch_lightning, cv2, trimesh, numpy
got = (torch.__version__, torch.cuda.is_available(), numpy.__version__)
print(*got)
assert got == ('2.0.1+cu118', True, '1.23.5'), 'expected 2.0.1+cu118 True 1.23.5'"
ls -lL "$ckpt" assets/hamer/data/mano/MANO_RIGHT.pkl
echo "annotation env ok"
