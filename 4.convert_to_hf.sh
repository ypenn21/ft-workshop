#!/bin/bash

# A script to convert a Gemma model to Hugging Face format.
# It sources configuration from a file.
#
# Usage: ./convert_to_vllm.sh [path/to/config_file.conf]
# If no config file is provided, it defaults to 'conversion.conf' in the same directory.

set -euo pipefail

# --- Configuration ---
CONFIG_FILE="${1:-config.conf}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found at '$CONFIG_FILE'" >&2
    exit 1
fi

source "$CONFIG_FILE"

# --- Execution ---
echo "Starting model conversion using config from '${CONFIG_FILE}'..."
python export_gemma_to_hf.py \
    --weights_file "${WEIGHTS_FILE}" \
    --size "${MODEL_SIZE}" \
    --vocab_path "${VOCAB_PATH}" \
    --gemma_version "${GEMMA_VERSION}" \
    --output_dir "${OUTPUT_DIR}"

echo "Conversion complete. Output is in '${OUTPUT_DIR}'."
