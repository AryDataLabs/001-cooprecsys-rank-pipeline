#!/usr/bin/env bash
# build_cython.sh — Compile _cy_predict.pyx Cython extension
set -euo pipefail
echo "🔧 Building Cython extensions..."
cd "$(dirname "$0")/.."
python _cy/setup.py build_ext --inplace
echo "✅ Cython build complete"
