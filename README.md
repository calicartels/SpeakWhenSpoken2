# meeting-ai

Real-time meeting perception pipeline.

```bash
bash run.sh
python audio_test.py
python sortformer.py
python ecapa.py
```

audio_test: LibriSpeech → test wav. Sortformer: diarization (80ms, 4 spk). ECAPA: embeddings (192-dim).
