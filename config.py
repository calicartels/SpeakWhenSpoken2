# config.py — paths, model names, streaming parameters

SAMPLE_RATE = 16000
TEST_AUDIO = "test_audio/two_speakers.wav"
HARD_AUDIO = "test_audio/hard_meeting.wav"
CLEAN_DIR = "test_audio/clean_sources"

SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"
SORTFORMER_CHUNK_LEN = 6
SORTFORMER_RIGHT_CONTEXT = 7
SORTFORMER_FIFO_LEN = 188
SORTFORMER_CACHE_UPDATE = 144
SORTFORMER_CACHE_LEN = 188

ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_EMB_DIM = 192

MIN_SEGMENT_SEC = 1.5
MAX_SEGMENT_SEC = 10.0
