#!/bin/bash
# run.sh — install dependencies on vast.ai (Ubuntu + CUDA + PyTorch pre-installed)
# Run once. Takes ~10-15 min mostly due to NeMo.
# Run from project root: bash run.sh

set -e
cd "$(dirname "$0")"

apt-get update && apt-get install -y libsndfile1 ffmpeg sox
pip install Cython packaging
pip install -r requirements.txt
pip install -r nemo_requirements.txt
pip install --no-deps 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'
pip install torchcodec
pip install 'speechbrain @ git+https://github.com/speechbrain/speechbrain.git@develop' datasets
mkdir -p test_audio

python -c "from nemo.collections.asr.models import SortformerEncLabelModel; print('NeMo OK')"
python -c "from speechbrain.inference.speaker import EncoderClassifier; print('SpeechBrain OK')"