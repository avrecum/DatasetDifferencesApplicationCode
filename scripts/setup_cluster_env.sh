#!/usr/bin/env bash
# Optional isolated model-library layer over an existing compatible Torch runtime.
set -euo pipefail
: "${PREFERENCE_BASE_PYTHON:?Set PREFERENCE_BASE_PYTHON to your existing Torch-enabled Python}"
"$PREFERENCE_BASE_PYTHON" -m venv .venv
PREFERENCE_BASE_SITE=$("$PREFERENCE_BASE_PYTHON" -c 'import site; print(site.getsitepackages()[0])')
export PREFERENCE_BASE_SITE
.venv/bin/python - <<'PY'
import os,site
from pathlib import Path
(Path(site.getsitepackages()[0])/'cluster_numeric_runtime.pth').write_text(os.environ['PREFERENCE_BASE_SITE']+'\n')
PY
.venv/bin/python -m pip install --ignore-installed --no-deps \
  transformers==4.57.6 tokenizers==0.22.2 huggingface-hub==0.36.2
.venv/bin/python -m pip install --no-deps -e .
