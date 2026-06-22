#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/opt/molink/hanging}"
CONDA_ENV="${CONDA_ENV:-metric3d}"
INSTALL_SAM="${INSTALL_SAM:-1}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"
SAM_DIR="${SAM_DIR:-/opt/molink-cache/sam}"
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$ROOT"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
if [ "$INSTALL_SAM" = "1" ]; then
  echo "[install] Installing Segment Anything adapter"
  python -m pip install -r requirements-sam.txt
  mkdir -p "$SAM_DIR"
  case "$SAM_MODEL_TYPE" in
    vit_b) ckpt="$SAM_DIR/sam_vit_b_01ec64.pth"; url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" ;;
    vit_l) ckpt="$SAM_DIR/sam_vit_l_0b3195.pth"; url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" ;;
    vit_h) ckpt="$SAM_DIR/sam_vit_h_4b8939.pth"; url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" ;;
    *) echo "Unsupported SAM_MODEL_TYPE=$SAM_MODEL_TYPE" >&2; exit 2 ;;
  esac
  if [ ! -f "$ckpt" ]; then
    echo "[install] Downloading $SAM_MODEL_TYPE checkpoint to $ckpt"
    if command -v wget >/dev/null 2>&1; then wget -O "$ckpt" "$url"; else curl -L "$url" -o "$ckpt"; fi
  else
    echo "[install] Found existing checkpoint: $ckpt"
  fi
  echo "export SAM_ENABLE=1"
  echo "export SAM_BACKEND=sam1"
  echo "export SAM_MODEL_TYPE=$SAM_MODEL_TYPE"
  echo "export SAM_CHECKPOINT=$ckpt"
fi
chmod +x start_*.sh scripts/*.sh 2>/dev/null || true
python - <<'PYCHK'
import importlib
for m in ['semantic_mvp.service', 'hanging_mvp.service']:
    importlib.import_module(m)
    print('[ok] import', m)
PYCHK
