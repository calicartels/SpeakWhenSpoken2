import './VapGauges.css';

const GATE_THRESHOLD = 0.55;

function Gauge({ label, value, color, max = 1 }) {
  const pct = Math.min(100, (value / max) * 100);
  const dashArray = 2 * Math.PI * 36;
  const dashOffset = dashArray * (1 - pct / 100);
  return (
    <div className="gauge">
      <svg viewBox="0 0 80 80" className="gauge__svg">
        <circle cx="40" cy="40" r="36" className="gauge__track" />
        <circle
          cx="40" cy="40" r="36"
          className="gauge__fill"
          style={{
            stroke: color,
            strokeDasharray: dashArray,
            strokeDashoffset: dashOffset,
          }}
        />
        <text x="40" y="38" className="gauge__value">{value.toFixed(2)}</text>
        <text x="40" y="52" className="gauge__label">{label}</text>
      </svg>
    </div>
  );
}

export default function VapGauges({ vap, state, gateOpen }) {
  const opening = vap?.ai_opening ?? 0;
  const hold = vap?.turn_hold ?? 0;
  const conf = vap?.confidence ?? 0;
  const silence = state?.silence_gap_sec ?? 0;

  const silenceSig = Math.min(silence / 2.0, 1.0);
  const domProb = state?.dominant_prob ?? 1;
  const fadeSig = Math.max(0, 1.0 - domProb / 0.5);
  const prosodic = 1.0 - hold;
  const composite = 0.35 * silenceSig + 0.30 * fadeSig + 0.20 * prosodic + 0.15 * opening;

  return (
    <div className="vap-gauges">
      <div className="vap-gauges__row">
        <Gauge label="AI Open" value={opening} color="var(--accent-indigo)" />
        <Gauge label="Hold" value={hold} color="var(--accent-amber)" />
        <Gauge label="Conf" value={conf} color="var(--accent-cyan)" />
      </div>

      <div className="vap-gauges__composite">
        <div className="vap-gauges__bar-label">
          <span>Gate Composite</span>
          <span className="vap-gauges__composite-val">{composite.toFixed(2)}</span>
        </div>
        <div className="vap-gauges__bar">
          <div
            className={`vap-gauges__bar-fill ${composite >= GATE_THRESHOLD ? 'active' : ''}`}
            style={{ width: `${Math.min(100, composite * 100)}%` }}
          />
          <div
            className="vap-gauges__threshold"
            style={{ left: `${GATE_THRESHOLD * 100}%` }}
          />
        </div>
      </div>

      {gateOpen && (
        <div className="vap-gauges__flash">
          ⚡ GATE OPEN
        </div>
      )}

      <div className="vap-gauges__meta">
        <span className="vap-gauges__meta-item">
          Mode: <strong>{state?.mode ?? '—'}</strong>
        </span>
        <span className="vap-gauges__meta-item">
          Silence: <strong>{silence.toFixed(1)}s</strong>
        </span>
        <span className="vap-gauges__meta-item">
          Dom: <strong>{state?.dominant ?? '—'}</strong>
        </span>
      </div>
    </div>
  );
}
