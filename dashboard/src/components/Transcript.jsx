import { useRef, useEffect } from 'react';
import './Transcript.css';

const SPEAKER_COLORS = ['#60a5fa', '#34d399', '#fbbf24', '#f472b6'];

export default function Transcript({ segments }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [segments.length]);

  if (segments.length === 0) {
    return (
      <div className="transcript transcript--empty">
        <span className="transcript__placeholder">Waiting for speech…</span>
      </div>
    );
  }

  return (
    <div className="transcript">
      {segments.map((seg, i) => {
        const color = SPEAKER_COLORS[(seg.slot_id ?? 0) % SPEAKER_COLORS.length];
        return (
          <div key={i} className="transcript__seg" style={{ animationDelay: `${i * 10}ms` }}>
            <span className="transcript__time">
              {seg.start?.toFixed(1)}s
            </span>
            <span className="transcript__speaker" style={{ color }}>
              {seg.speaker || `slot_${seg.slot_id}`}
            </span>
            <span className="transcript__text">{seg.text}</span>
          </div>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}
