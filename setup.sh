sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
