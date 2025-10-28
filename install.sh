# Use Python 3.11 (TensorFlow 2.16+ supports 3.11; 3.12 can be tricky)
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 1) Install TensorFlow for Apple Silicon
pip install "tensorflow-macos>=2.16,<2.20" "tensorflow-metal>=1.1"

# 2) Install RetinaFace WITHOUT auto-deps (to avoid TF 2.5)
pip install --no-deps "retina-face>=0.0.17"

# 3) Rest of your stack
pip install "PyQt5==5.15.11" "opencv-python>=4.10" "numpy>=1.26" "Pillow>=10.4"
