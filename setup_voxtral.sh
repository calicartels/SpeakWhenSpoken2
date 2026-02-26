#!/bin/bash
# setup_voxtral.sh — Voxtral Realtime deps (transformers v5)
# Run after run.sh. Upgrading transformers may break NeMo.

set -e
cd "$(dirname "$0")"

pip install --upgrade "transformers>=5.0"
pip install mistral_common accelerate

python -c "from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor; print('Voxtral OK')"
