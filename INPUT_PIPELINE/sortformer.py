"""Streaming Sortformer diarization — frame-by-frame, no batching.

Adapted from NVIDIA NeMo's streaming diarizer:
https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/agents/voice_agent/pipecat/services/nemo/streaming_diar.py

The diarizer processes each incoming audio chunk (~0.08-0.13s) through
a sliding feature buffer + the Sortformer forward_streaming_step, returning
per-frame speaker probabilities with sub-second latency.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules.sortformer_modules import StreamingSortformerState
from omegaconf import DictConfig

import config

LOG_MEL_ZERO = -16.635
FRAME_SEC = 0.08


class _AudioBuffer:
    def __init__(self, sample_rate, buffer_sec):
        self.size = int(buffer_sec * sample_rate)
        self.buf = torch.zeros(self.size, dtype=torch.float32)

    def reset(self):
        self.buf.zero_()

    def update(self, audio):
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        n = audio.shape[0]
        self.buf[:-n] = self.buf[n:].clone()
        self.buf[-n:] = audio.clone()

    def empty(self):
        return self.buf.sum() == 0


class _FeatureBuffer:
    def __init__(self, sample_rate, buffer_sec, chunk_sec, preprocessor_cfg, device):
        self.sample_rate = sample_rate
        self.device = device
        self.n_feat = preprocessor_cfg.features
        self.timestep = preprocessor_cfg.window_stride
        self.lookback = int(self.timestep * sample_rate)
        self.chunk_samples = int(chunk_sec * sample_rate)
        self.audio = _AudioBuffer(sample_rate, buffer_sec)

        self.feat_buf_len = int(buffer_sec / self.timestep)
        self.feat_chunk_len = int(chunk_sec / self.timestep)
        self.feat_buf = torch.full(
            [self.n_feat, self.feat_buf_len], LOG_MEL_ZERO,
            dtype=torch.float32, device=device,
        )

        self.preprocessor = nemo_asr.models.ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(device)

    def reset(self):
        self.audio.reset()
        self.feat_buf.fill_(LOG_MEL_ZERO)

    def update(self, audio_f32):
        self.audio.update(audio_f32)
        if math.isclose(self.audio.size / self.sample_rate,
                        self.chunk_samples / self.sample_rate, rel_tol=0.01):
            samples = self.audio.buf.clone()
        else:
            samples = self.audio.buf[-(self.lookback + self.chunk_samples):]

        sig = samples.unsqueeze(0).to(self.device)
        sig_len = torch.tensor([sig.shape[1]], device=self.device)
        feats, _ = self.preprocessor(input_signal=sig, length=sig_len)
        feats = feats.squeeze()

        diff = feats.shape[1] - self.feat_chunk_len - 1
        if diff > 0:
            feats = feats[:, :-diff]

        self.feat_buf[:, :-self.feat_chunk_len] = self.feat_buf[:, self.feat_chunk_len:].clone()
        self.feat_buf[:, -self.feat_chunk_len:] = feats[:, -self.feat_chunk_len:]

    def get(self):
        return self.feat_buf.clone()


@dataclass
class _Cfg:
    max_speakers: int = 4
    chunk_len: int = config.SORTFORMER_CHUNK_LEN
    chunk_left_context: int = 1
    chunk_right_context: int = config.SORTFORMER_RIGHT_CONTEXT
    fifo_len: int = config.SORTFORMER_FIFO_LEN
    spkcache_len: int = config.SORTFORMER_CACHE_LEN
    spkcache_refresh: int = config.SORTFORMER_CACHE_UPDATE
    device: str = "cuda"


class _StreamDiarizer:
    def __init__(self, cfg, model_name):
        self.cfg = cfg
        self.device = cfg.device
        self.chunk_size = cfg.chunk_len
        self.max_spk = cfg.max_speakers
        self.left_off = 8
        self.right_off = 8

        self.model = SortformerEncLabelModel.from_pretrained(
            model_name, map_location=cfg.device,
        )
        self.model.sortformer_modules.chunk_len = cfg.chunk_len
        self.model.sortformer_modules.spkcache_len = cfg.spkcache_len
        self.model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
        self.model.sortformer_modules.chunk_right_context = cfg.chunk_right_context
        self.model.sortformer_modules.fifo_len = cfg.fifo_len

        if hasattr(self.model.sortformer_modules, 'spkcache_refresh_rate'):
            self.model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh
        elif hasattr(self.model.sortformer_modules, 'spkcache_update_period'):
            self.model.sortformer_modules.spkcache_update_period = cfg.spkcache_refresh

        if hasattr(self.model.sortformer_modules, 'log'):
            self.model.sortformer_modules.log = False
        self.model.sortformer_modules._check_streaming_parameters()
        self.model.eval()

        buf_sec = cfg.chunk_len * FRAME_SEC + (self.left_off + self.right_off) * 0.01
        chunk_sec = cfg.chunk_len * FRAME_SEC

        self.feat_buf = _FeatureBuffer(
            sample_rate=config.SAMPLE_RATE,
            buffer_sec=buf_sec,
            chunk_sec=chunk_sec,
            preprocessor_cfg=self.model.cfg.preprocessor,
            device=self.device,
        )
        self.state = self._init_state()
        self.total_preds = torch.zeros((1, 0, self.max_spk), device=self.device)

    def _init_state(self):
        async_flag = getattr(self.model, 'async_streaming', False)
        return self.model.sortformer_modules.init_streaming_state(
            batch_size=1, async_streaming=async_flag, device=self.device,
        )

    def push(self, audio_f32):
        """Push float32 audio samples, return [chunk_len, max_spk] probs."""
        self.feat_buf.update(audio_f32)
        feats = self.feat_buf.get().unsqueeze(0).transpose(1, 2)
        feat_len = torch.tensor([feats.shape[1]], device=self.device)

        with torch.inference_mode(), torch.no_grad():
            self.state, preds = self.model.forward_streaming_step(
                processed_signal=feats,
                processed_signal_length=feat_len,
                streaming_state=self.state,
                total_preds=self.total_preds,
                left_offset=self.left_off,
                right_offset=self.right_off,
            )

        self.total_preds = preds
        chunk = preds[:, -self.chunk_size:, :].cpu().numpy()
        return chunk[0]

    def reset(self):
        self.feat_buf.reset()
        self.state = self._init_state()
        self.total_preds = torch.zeros((1, 0, self.max_spk), device=self.device)


# --- Public API ---

def load_model():
    cfg = _Cfg()
    diarizer = _StreamDiarizer(cfg, config.SORTFORMER_MODEL)
    return diarizer


def push_audio(diarizer, samples_f32):
    """Push audio chunk, get back list of [4] prob arrays (one per frame)."""
    result = diarizer.push(samples_f32)
    return [list(result[t]) for t in range(result.shape[0])]


def reset(diarizer):
    diarizer.reset()
