import { useRef, useEffect } from 'react';
import './SpeakerWaveforms.css';

const CANVAS_W = 400;
const CANVAS_H = 80;

export default function SpeakerWaveforms({ frameData }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const samples = frameData?.current;
    const canvas = canvasRef.current;
    if (!samples || samples.length < 2 || !canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#dd5e4a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const step = Math.max(1, Math.floor(samples.length / w));
    const mid = h / 2;
    
    for (let x = 0; x < w; x++) {
      const idx = Math.min(x * step, samples.length - 1);
      const v = samples[idx] || 0;
      const y = mid - v * mid * 4;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  return (
    <div className="waveforms">
      <div className="waveforms__row">
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          className="waveforms__canvas single-waveform"
        />
      </div>
    </div>
  );
}
