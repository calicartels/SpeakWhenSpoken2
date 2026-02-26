# meeting-ai

Real-time meeting perception pipeline.

```bash
bash run.sh
python tests/audio_test.py
python sortformer.py
python ecapa.py
```

Hard tests (crosstalk, rapid turns, speaker return):

```bash
python tests/hard_audio.py
python tests/hard_diar.py
python tests/hard_identity.py
```

Voxtral (optional, transformers v5 may conflict with NeMo):

```bash
bash setup_voxtral.sh
python voxtral.py
```

audio_test: LibriSpeech → test wav. Sortformer: diarization (80ms, 4 spk). ECAPA: embeddings (192-dim). Voxtral: streaming ASR.
