# The order is important
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
pip install jax[tpu] tensorflow-cpu keras-hub sentencepiece google-cloud-aiplatform
pip install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
