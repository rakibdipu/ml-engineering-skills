#!/usr/bin/env bash
# One-command installer for ML Engineering Skills Plugin
# Author: MD. RaKib Hassan Dipu (https://github.com/rakibdipu)

set -e

PLUGIN_NAME="ml-engineering"
TARGET_DIR="${HOME}/.gemini/config/plugins/${PLUGIN_NAME}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================================="
echo " Installing ML Engineering Skills Plugin"
echo " Author: MD. RaKib Hassan Dipu (@rakibdipu)"
echo "=========================================================="

mkdir -p "${TARGET_DIR}/skills"

echo "Copying plugin manifest..."
cp "${SRC_DIR}/plugin.json" "${TARGET_DIR}/"

echo "Installing skills: timesfm-forecasting, tabular-scarf, airflow-ai-orchestration..."
cp -r "${SRC_DIR}/skills/"* "${TARGET_DIR}/skills/"

echo ""
echo "Installation complete!"
echo "Destination: ${TARGET_DIR}"
echo ""
echo "Available Skills:"
echo "  1. timesfm-forecasting       - Google TimesFM foundation model forecasting"
echo "  2. tabular-scarf             - Self-supervised contrastive learning for tabular data"
echo "  3. airflow-ai-orchestration  - Airflow AI SDK & common-ai TaskFlow orchestration"
echo "=========================================================="
