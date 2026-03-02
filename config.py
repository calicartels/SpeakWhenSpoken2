SAMPLE_RATE = 16000
CHUNK_MS = 80
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHANNELS = 1

WS_HOST = "0.0.0.0"
WS_PORT = 8765

SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"
SORTFORMER_CHUNK_LEN = 6
SORTFORMER_RIGHT_CONTEXT = 7
SORTFORMER_FIFO_LEN = 188
SORTFORMER_CACHE_UPDATE = 144
SORTFORMER_CACHE_LEN = 188

VOXTRAL_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
VLLM_PORT = 8000

GLINER_MODEL = "knowledgator/gliner-multitask-large-v0.5"
ENTITY_LABELS = [
    "person", "organization", "product", "date", "decision",
    "action item", "deadline", "topic", "project name",
    "person owns deadline", "decision about topic",
    "commitment by person",
]
ENTITY_THRESHOLD = 0.4

GATE_THRESHOLD = 0.70
GATE_SILENCE_MIN = 1.5
GATE_SILENCE_FORCE = 3.0
GATE_SUPPRESS_SEC = 3.0
GATE_COOLDOWN_SEC = 5.0

MERCURY_API_URL = "https://api.inceptionlabs.ai/v1/chat/completions"
MERCURY_MODEL = "mercury-2"

PREWARM_THRESHOLD = 0.4
DRAFT_STALE_SEC = 10.0

WAKE_PHRASES = ["hey fauna", "fauna", "assistant", "hey assistant"]
WAKE_WORD_WINDOW = 5

SUPERMEMORY_ENABLED = True
CROSS_MEETING_SEARCH_LIMIT = 8

TTS_ENABLED = True
TTS_VOICE = "Rachel"
TTS_MODEL = "eleven_turbo_v2_5"

SEGMENT_ACTIVE_THRESHOLD = 0.3
SEGMENT_MERGE_GAP_FRAMES = 5
SEGMENT_MIN_SEC = 0.5
MIN_SEGMENT_SEC = 1.5
MAX_SEGMENT_SEC = 10.0

ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_EMB_DIM = 192
IDENTITY_MIN_SEC = 2.0
IDENTITY_THRESHOLD = 0.55
ADDRESS_BOOK_PATH = "state/address_book.pt"
FIFTH_SPEAKER_THRESHOLD = 0.40

PROBS_CACHE_PATH = "assets/sortformer_probs.npy"
TRANSCRIPT_CACHE_PATH = "assets/transcript_cache.json"
HIGH_OPENING_THRESHOLD = 0.75
TRANSCRIPT_WINDOW_SEC = 5.0

DECISION_MODEL = "anthropic/claude-haiku-4.5"
DECISION_USE_NITRO = False
DECISION_MAX_TOKENS = 200
DECISION_TEMPERATURE = 0.0

TEST_AUDIO = "test_audio/two_speakers.wav"
HARD_AUDIO = "test_audio/hard_meeting.wav"
TEST_AUDIO_ASSET = "assets/Test_audio_20min.wav"
TEST_AUDIO_2MIN = "assets/Test_audio_2min.wav"
TEST_AUDIO_20MIN = "assets/Test_audio_20min.wav"
CLEAN_DIR = "test_audio/clean_sources"
