SAMPLE_RATE = 16000
TEST_AUDIO = "test_audio/two_speakers.wav"
HARD_AUDIO = "test_audio/hard_meeting.wav"
TEST_AUDIO_ASSET = "assets/Test_audio_20min.wav"
TEST_AUDIO_2MIN = "assets/Test_audio_2min.wav"
TEST_AUDIO_20MIN = "assets/Test_audio_20min.wav"
CLEAN_DIR = "test_audio/clean_sources"

SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"
SORTFORMER_CHUNK_LEN = 6
SORTFORMER_RIGHT_CONTEXT = 7
SORTFORMER_FIFO_LEN = 188
SORTFORMER_CACHE_UPDATE = 144
SORTFORMER_CACHE_LEN = 188

ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_EMB_DIM = 192

VOXTRAL_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"

MIN_SEGMENT_SEC = 1.5
MAX_SEGMENT_SEC = 10.0

IDENTITY_MIN_SEC = 2.0
IDENTITY_THRESHOLD = 0.55
ADDRESS_BOOK_PATH = "state/address_book.pt"
FIFTH_SPEAKER_THRESHOLD = 0.40

PROBS_CACHE_PATH = "assets/sortformer_probs.npy"
TRANSCRIPT_CACHE_PATH = "assets/transcript_cache.json"
HIGH_OPENING_THRESHOLD = 0.75
TRANSCRIPT_WINDOW_SEC = 5.0

GATE_THRESHOLD_SILENCE = 0.85
GATE_THRESHOLD_SPEECH = 0.90
GATE_FADE_PROB = 0.5
GATE_SUPPRESS_SEC = 3.0

SEGMENT_ACTIVE_THRESHOLD = 0.3
SEGMENT_MERGE_GAP_FRAMES = 5
SEGMENT_MIN_SEC = 0.5

GLINER_MODEL = "urchade/gliner_medium-v2.1"
ENTITY_LABELS = ["person", "company", "product", "topic", "date", "number", "action item", "decision"]
ENTITY_THRESHOLD = 0.4

DECISION_MODEL = "meta-llama/llama-4-scout"
DECISION_USE_NITRO = True
DECISION_MAX_TOKENS = 200
DECISION_TEMPERATURE = 0.1
