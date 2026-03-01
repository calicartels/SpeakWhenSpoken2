import { useRef, useEffect } from 'react';
import './FaunaChat.css';
import './FaunaChat.css';

export default function FaunaChat({ messages, liveDraft }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  if (messages.length === 0 && !liveDraft?.text && !liveDraft?.generating) {
    return (
      <div className="fauna-chat fauna-chat--empty">
        <div className="fauna-chat__avatar">🦎</div>
        <span className="fauna-chat__placeholder">Fauna is listening…</span>
      </div>
    );
  }

  return (
    <div className="fauna-chat">
      {messages.map((msg, i) => (
        <div key={i} className="fauna-chat__bubble" style={{ animationDelay: `${i * 30}ms` }}>
          <div className="fauna-chat__header">
            <span className="fauna-chat__avatar-sm">🦎</span>
            <span className={`fauna-chat__tag fauna-chat__tag--${msg.source}`}>
              {msg.source}
            </span>
            <span className="fauna-chat__ts">{msg.timestamp?.toFixed(1)}s</span>
          </div>
          <div className="fauna-chat__text">{msg.text}</div>
          {msg.ai_opening && (
            <div className="fauna-chat__meta">
              VAP: {msg.ai_opening?.toFixed(2)} · {msg.mode ?? ''}
            </div>
          )}
        </div>
      ))}
      
      {(liveDraft?.generating || liveDraft?.text) && (
        <div className="fauna-chat__bubble fauna-chat__bubble--draft">
          <div className="fauna-chat__header">
            <span className="fauna-chat__avatar-sm" style={{ opacity: 0.5 }}>💭</span>
            <span className="fauna-chat__tag fauna-chat__tag--draft" style={{ background: 'var(--accent-amber-dim)', color: 'var(--accent-amber)' }}>
              live inner monologue
            </span>
            {liveDraft.generating && <span className="fauna-chat__ts">generating...</span>}
          </div>
          <div className="fauna-chat__text" style={{ fontStyle: 'italic', color: 'var(--text-secondary)' }}>
            {liveDraft.text ? liveDraft.text : "Formulating response based on context..."}
          </div>
        </div>
      )}

      <div ref={endRef} />
    </div>
  );
}
