#!/bin/bash
#
# This script sets up the environment by installing necessary packages,
# configuring system settings, and mounting a GCS bucket.
#
# It also cleans up redundant repository setup commands and adds robust
# error handling.
#
# Usage: ./setup.sh <your-gcs-bucket-name>

set -euo pipefail

# --- Configuration ---
BUCKET_NAME="${1:-}"

if [[ -z "$BUCKET_NAME" ]]; then
    echo "❌ Error: Bucket name must be provided as the first argument." >&2
    echo "Usage: ./setup.sh <your-gcs-bucket-name>" >&2
    exit 1
fi

echo "--- Setting up environment for GCS Bucket: ${BUCKET_NAME} ---"

# --- System Configuration ---
echo "⚙️  Configuring transparent huge pages..."
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

# --- GCS Fuse Installation (Modern and de-duplicated) ---
echo "⚙️  Installing GCS Fuse..."
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc > /dev/null
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null
sudo apt-get update
sudo apt-get install -y gcsfuse

# --- Mount GCS Bucket ---
echo "⚙️  Mounting GCS bucket..."
mkdir -p content
gcsfuse --implicit-dirs "${BUCKET_NAME}" content
echo "✔️ Bucket '${BUCKET_NAME}' mounted to './content'."

# --- Python Dependencies ---
echo "⚙️  Installing Python packages..."
pip install jax[tpu] tensorflow-cpu keras-hub sentencepiece google-cloud-aiplatform
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate

echo "--- ✅ Setup complete ---"
