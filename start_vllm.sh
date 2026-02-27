#!/usr/bin/env bash
# Start vLLM serving Voxtral Realtime on port 8000.
# Uses an isolated venv so NeMo/Sortformer deps don't conflict.
set -e

VENV=/workspace/vllm-env
PORT=${VLLM_PORT:-8000}
MODEL="mistralai/Voxtral-Mini-4B-Realtime-2602"

if [ ! -d "$VENV" ]; then
    echo "Creating vLLM venv at $VENV..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    echo "Installing vLLM (this takes a few minutes)..."
    "$VENV/bin/pip" install vllm soundfile
fi

echo "Starting vLLM serve on port $PORT..."
exec "$VENV/bin/vllm" serve "$MODEL" \
    --enforce-eager \
    --port "$PORT" \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --trust-remote-code \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.60
