# meeting-ai

Real-time meeting perception pipeline.

```bash
bash run.sh
python make_test_audio.py
python sortformer.py
python ecapa.py
```

Sortformer: diarization (80ms frames, 4 speakers). ECAPA-TDNN: embeddings (192-dim).
