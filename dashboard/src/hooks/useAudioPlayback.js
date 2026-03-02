import { useRef, useCallback } from 'react';

export function useAudioPlayback() {
  const ctxRef = useRef(null);
  const playingRef = useRef(false);

  const getCtx = useCallback(() => {
    if (!ctxRef.current) {
      ctxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    return ctxRef.current;
  }, []);

  const play = useCallback(async (b64) => {
    if (!b64 || playingRef.current) return;
    playingRef.current = true;
    const ctx = getCtx();
    if (ctx.state === 'suspended') await ctx.resume();

    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

    const buf = await ctx.decodeAudioData(bytes.buffer);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.onended = () => { playingRef.current = false; };
    src.start(0);
  }, [getCtx]);

  return { play, playing: playingRef };
}
