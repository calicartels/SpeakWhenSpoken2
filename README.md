# SpeakWhenSpoken2

Can you put an AI voice agent into a 4-person meeting and have it actually work?

<video src="https://github.com/calicartels/SpeakWhenSpoken2/blob/main/assets/video.mp4" controls width="100%"></video>

Inspired by [Ishiki Labs'](https://twitter.com/ishikilabs) Fern (YC W26) — except this one focuses on group conversations. Not 1-on-1 voice assistants (those are solved). Group settings are fundamentally harder: overlapping speakers, fragmented context, backchannels misread as turn endings, and silence that belongs to someone thinking rather than open floor.

Built on Voice Activity Projection, Streaming Sortformer diarization, and Mercury 2 (1000+ tok/sec diffusion LLM). Runs on a single rented 3090 for under $10 in compute.

[Full writeup →](https://x.com/vishnutm244412/status/2028279537717432717)

![Architecture](https://github.com/calicartels/SpeakWhenSpoken2/blob/main/assets/image.png)



---

## Setup

### Environment

Create a `.env` file in the project root:

```
MERCURY_API_KEY=
ELEVENLABS_API_KEY=
SUPERMEMORY_API_KEY=
```

### Server (Vast.ai 3090)

```bash
ssh -p <PORT> root@<IP>
cd /workspace/SpeakWhenSpoken2

pip install -r requirements.txt
pip install nemo_toolkit[asr]
pip install vllm --no-deps

python -c "import nemo; import vllm; import gliner; print('ok')"
```

Start vLLM (Voxtral Realtime):

```bash
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --served-model-name voxtral \
  --max-model-len 32768 \
  --port 8000 &
```

Wait for vLLM to be healthy, then start the pipeline:

```bash
python server.py
```

Server listens on `ws://0.0.0.0:8765`.

### Local (Mac)

```bash
pip install -r requirements-client.txt
cd dashboard && npm install
```

SSH tunnel to forward the WebSocket:

```bash
ssh -p <PORT> root@<IP> -L 8765:localhost:8765 -N
```

Start the dashboard:

```bash
cd dashboard
npm run dev
```

Opens at `http://localhost:5173`. Click **Connect**, then use mic or stream a YouTube video into the pipeline.

---

## References

1. Ekstedt & Skantze, "Voice Activity Projection," Interspeech 2022
2. Inoue et al., "Real-time Turn-taking Prediction Using VAP," IWSDS 2024
3. Inoue et al., "Multilingual Turn-taking Prediction," LREC-COLING 2024
4. Elmers et al., "Triadic Multi-party VAP," Interspeech 2025
5. Medennikov et al., "Streaming Sortformer," 2025
6. Park, Medennikov et al., "Sortformer," 2024
7. Desplanques et al., "ECAPA-TDNN," Interspeech 2020
8. MaAI-Kyoto, [MaAI](https://github.com/MaAI-Kyoto/MaAI)
9. Mistral AI, "Voxtral Realtime," 2026
10. Zaratiana et al., "GLiNER," 2023
11. Inception Labs, [Mercury](https://inceptionlabs.ai)
