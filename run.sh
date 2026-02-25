#!/bin/bash
# run.sh — install dependencies on vast.ai (Ubuntu + CUDA + PyTorch pre-installed)
# Run once. Takes ~10-15 min mostly due to NeMo.

set -e

echo "Installing system deps..."
apt-get update && apt-get install -y libsndfile1 ffmpeg sox

echo "Installing pinned deps FIRST (prevents NeMo from overwriting)..."
pip install Cython packaging
pip install -r requirements.txt

echo "Installing NeMo (--no-deps to respect pinned versions)..."
pip install --no-deps 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'

echo "Installing remaining NeMo sub-deps..."
pip install hydra-core omegaconf lightning torchmetrics editdistance \
    jiwer sentencepiece transformers librosa webdataset lhotse braceexpand \
    nv-one-logger-core nv-one-logger-training-telemetry \
    nv-one-logger-pytorch-lightning-integration

echo "Installing SpeechBrain + datasets..."
pip install speechbrain datasets

mkdir -p test_audio

echo "Verifying imports..."
python -c "from nemo.collections.asr.models import SortformerEncLabelModel; print('NeMo OK')"
python -c "from speechbrain.inference.speaker import EncoderClassifier; print('SpeechBrain OK')"

echo "Done. Run: python make_test_audio.py"