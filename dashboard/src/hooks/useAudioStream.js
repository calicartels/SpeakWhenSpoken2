import { useRef, useCallback, useState } from 'react';

const TARGET_RATE = 16000;
const CHUNK_SAMPLES = 1280; // 80ms at 16kHz

// Proper ArrayBuffer to base64
function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunks = [];
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    chunks.push(String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize)));
  }
  return btoa(chunks.join(''));
}

// Simple linear downsample from srcRate to TARGET_RATE
function downsample(buffer, srcRate) {
  if (srcRate === TARGET_RATE) return buffer;
  const ratio = srcRate / TARGET_RATE;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    result[i] = buffer[Math.round(i * ratio)];
  }
  return result;
}

export function useAudioStream(sendFn) {
  const [active, setActive] = useState(false);
  const [source, setSource] = useState(null);
  const ctxRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const frameDataRef = useRef(new Float32Array(0));
  const sourceNodeRef = useRef(null);
  const chunkCountRef = useRef(0);
  const residualRef = useRef(new Float32Array(0));

  const sendChunks = useCallback((samples) => {
    // Accumulate residual + new samples, then send CHUNK_SAMPLES-sized pieces
    const combined = new Float32Array(residualRef.current.length + samples.length);
    combined.set(residualRef.current);
    combined.set(samples, residualRef.current.length);

    let offset = 0;
    while (offset + CHUNK_SAMPLES <= combined.length) {
      const chunk = combined.slice(offset, offset + CHUNK_SAMPLES);
      offset += CHUNK_SAMPLES;

      chunkCountRef.current++;
      if (chunkCountRef.current <= 3 || chunkCountRef.current % 200 === 0) {
        const maxVal = Math.max(...Array.from(chunk.slice(0, 50)).map(Math.abs));
        console.log(`[audio] chunk #${chunkCountRef.current}, max=${maxVal.toFixed(4)}`);
      }

      const encoded = arrayBufferToBase64(chunk.buffer);
      sendFn(JSON.stringify({ type: 'audio', data: encoded }));
    }

    residualRef.current = combined.slice(offset);

    // Store for waveform visualization (keep last 2s)
    const prev = frameDataRef.current;
    const merged = new Float32Array(prev.length + samples.length);
    merged.set(prev);
    merged.set(samples, prev.length);
    frameDataRef.current = merged.length > TARGET_RATE * 2
      ? merged.slice(-TARGET_RATE * 2) : merged;
  }, [sendFn]);

  const startMic = useCallback(async () => {
    console.log('[audio] startMic called');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1 }
      });
      console.log('[audio] got mic stream');

      // Use DEFAULT sample rate (usually 48kHz) — don't force 16kHz
      const ctx = new AudioContext();
      await ctx.resume();
      const nativeRate = ctx.sampleRate;
      console.log('[audio] AudioContext native rate:', nativeRate);

      const src = ctx.createMediaStreamSource(stream);
      // Buffer size 4096 works reliably across browsers
      const proc = ctx.createScriptProcessor(4096, 1, 1);

      chunkCountRef.current = 0;
      residualRef.current = new Float32Array(0);

      proc.onaudioprocess = (e) => {
        const raw = e.inputBuffer.getChannelData(0);
        const resampled = downsample(new Float32Array(raw), nativeRate);
        sendChunks(resampled);
      };

      src.connect(proc);
      proc.connect(ctx.destination);

      ctxRef.current = ctx;
      streamRef.current = stream;
      processorRef.current = proc;
      setActive(true);
      setSource('mic');
      console.log('[audio] mic streaming active');
    } catch (err) {
      console.error('[audio] Mic error:', err);
    }
  }, [sendChunks]);

  const startVideo = useCallback(async (videoElement) => {
    console.log('[audio] startVideo called');
    try {
      if (sourceNodeRef.current) {
        console.log('[audio] already connected');
        return;
      }

      const ctx = new AudioContext();
      await ctx.resume();
      const nativeRate = ctx.sampleRate;
      console.log('[audio] video AudioContext native rate:', nativeRate);

      const src = ctx.createMediaElementSource(videoElement);
      const proc = ctx.createScriptProcessor(4096, 1, 1);

      chunkCountRef.current = 0;
      residualRef.current = new Float32Array(0);

      proc.onaudioprocess = (e) => {
        const raw = e.inputBuffer.getChannelData(0);
        const resampled = downsample(new Float32Array(raw), nativeRate);
        sendChunks(resampled);
      };

      src.connect(proc);
      proc.connect(ctx.destination);
      src.connect(ctx.destination);

      ctxRef.current = ctx;
      processorRef.current = proc;
      sourceNodeRef.current = src;
      setActive(true);
      setSource('video');
      console.log('[audio] video streaming active');

      videoElement.play().catch(err => console.warn('[audio] play failed:', err));
    } catch (err) {
      console.error('[audio] Video error:', err);
    }
  }, [sendChunks]);

  const stop = useCallback(() => {
    console.log('[audio] stopping');
    if (processorRef.current) { try { processorRef.current.disconnect(); } catch (e) { void e; } }
    if (sourceNodeRef.current) { try { sourceNodeRef.current.disconnect(); } catch (e) { void e; } }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); }
    if (ctxRef.current) { try { ctxRef.current.close(); } catch (e) { void e; } }
    processorRef.current = null;
    sourceNodeRef.current = null;
    streamRef.current = null;
    ctxRef.current = null;
    residualRef.current = new Float32Array(0);
    frameDataRef.current = new Float32Array(0);
    setActive(false);
    setSource(null);
  }, []);

  return { active, source, startMic, startVideo, stop, frameDataRef };
}
