#!/bin/bash
# Fauna Pipeline — One-command startup
# Usage: bash start.sh
# Run this in the Vast.ai Jupyter terminal

set -e
cd /workspace/SpeakWhenSpoken2

echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "⚠ nvidia-smi not found"

echo "🔍 Checking vLLM..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✅ vLLM already running"
else
    echo "🚀 Starting vLLM (takes ~90s to load model)..."
    python -m vllm.entrypoints.openai.api_server \
        --model mistralai/Voxtral-Mini-4B-Realtime-2602 \
        --port 8000 --dtype float16 --enforce-eager &> vllm.log &
    VLLM_PID=$!
    echo "   PID: $VLLM_PID — waiting for model to load..."

    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "✅ vLLM ready after $((i * 2))s"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "❌ vLLM crashed. Check: tail -30 vllm.log"
            exit 1
        fi
        sleep 2
    done

    if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "❌ vLLM timed out after 120s. Check: tail -30 vllm.log"
        exit 1
    fi
fi

echo "🎯 Starting pipeline server..."
echo "   WebSocket: ws://0.0.0.0:8765"
echo "   Stop with Ctrl+C"
echo ""
python server.py
