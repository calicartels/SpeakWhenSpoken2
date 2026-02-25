#!/bin/bash
# run.sh — install dependencies on vast.ai (Ubuntu + CUDA + PyTorch pre-installed)
# Run once. Takes ~10-15 min mostly due to NeMo.

set -e

echo "Installing system deps..."
apt-get update && apt-get install -y libsndfile1 ffmpeg sox

echo "Installing pinned deps FIRST (prevents NeMo from overwriting)..."
pip install Cython packaging
pip install -r requirements.txt

echo "Installing NeMo ASR dependencies..."
pip install -r nemo_requirements.txt

echo "Installing NeMo (--no-deps to respect pinned versions)..."
pip install --no-deps 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'

echo "Installing SpeechBrain + datasets..."
# SpeechBrain develop: fixes torchaudio 2.9+ (list_audio_backends removed)
pip install 'speechbrain @ git+https://github.com/speechbrain/speechbrain.git@develop' datasets

mkdir -p test_audio

echo "Verifying imports..."
python -c "from nemo.collections.asr.models import SortformerEncLabelModel; print('NeMo OK')"
python -c "from speechbrain.inference.speaker import EncoderClassifier; print('SpeechBrain OK')"

echo "Done. Run: python make_test_audio.py"