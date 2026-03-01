# Fauna Pipeline — Startup Guide

## Quick Reference

| Component | Where | Command |
|---|---|---|
| vLLM + server.py | Vast.ai Jupyter terminal | `bash start.sh` |
| SSH tunnel | Mac terminal 1 | `ssh -p 42345 root@109.173.238.203 -L 8765:localhost:8765 -N` |
| Dashboard | Mac terminal 2 | `cd dashboard && npm run dev` |
| YouTube server | Mac terminal 3 | `cd dashboard && npm run yt` |

## Step-by-Step

### 1. Start the server (Vast.ai Jupyter terminal)

```bash
cd /workspace/SpeakWhenSpoken2
bash start.sh
```

This automatically:
- Checks if vLLM is running (skips if yes)
- Starts vLLM and waits ~90s for it to load
- Starts `server.py` (Sortformer + VAP + GLiNER)

Wait until you see: `Server listening on ws://0.0.0.0:8765`

### 2. SSH tunnel (Mac terminal 1)

```bash
ssh -p 42345 root@109.173.238.203 -L 8765:localhost:8765 -N
```

`-N` means "no shell, just tunnel." Leave this running.

### 3. Dashboard (Mac terminal 2)

```bash
cd "/Users/vishnumukundan/Documents/Duke Code/SS2/SpeakWhenSpoken2/dashboard"
npm run dev
```

Opens at http://localhost:5173

### 4. YouTube download server (Mac terminal 3, optional)

```bash
cd "/Users/vishnumukundan/Documents/Duke Code/SS2/SpeakWhenSpoken2/dashboard"
npm run yt
```

### 5. Use the dashboard

1. Open http://localhost:5173
2. Click **Connect** → badge turns green
3. **Microphone**: Click 🎙 Mic → speak → watch Frames go up
4. **YouTube video**: Paste URL → Download → Play → click "🔊 Stream Audio to Pipeline"
5. **Cached video**: Type `/downloads/fGKNUvivvnc.mp4` → Load → Play → Stream

## Environment Setup (first time or after reset)

On the Vast.ai Jupyter terminal:

```bash
cd /workspace/SpeakWhenSpoken2
pip install "protobuf~=5.29.5" --force-reinstall
pip install vllm --no-deps
pip install -r requirements.txt
pip install -r nemo_requirements.txt
```

Verify:
```bash
python -c "import nemo; import vllm; import gliner; print('All good')"
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `vLLM not ready` | vLLM crashed — restart with `bash start.sh` |
| `Bearer ` error | Missing `.env` on server — copy it to `/workspace/SpeakWhenSpoken2/.env` |
| Frames stuck at 0 | Check browser console (Cmd+Opt+J) for errors |
| `channel N: open failed` | Server not running — start it first, then SSH |
| Video won't play | Load from `/downloads/filename.mp4` (same origin) |
