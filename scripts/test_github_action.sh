#!/usr/bin/env bash
set -euo pipefail

# Local rehearsal of the GitHub Action build:
# 1) Ensure typst + python3 are installed
# 2) Run the site build script

echo "Typst version: $(typst --version)"
echo "Python version: $(python3 --version)"

python3 scripts/build_site.py

echo "Build artifacts ready in ./build (mirrors workflow output)."
