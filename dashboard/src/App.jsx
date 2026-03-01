import { useState, useEffect, useCallback, useRef } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useWebSocket } from './hooks/useWebSocket';
import { useAudioStream } from './hooks/useAudioStream';

import Layout from './components/Layout';
import Home from './pages/Home';
import Intelligence from './pages/Intelligence';
import Memory from './pages/Memory';

import './App.css';

// Read from env or default
const SM_API_KEY = import.meta.env.VITE_SUPERMEMORY_API_KEY || '';
const DEFAULT_WS = 'ws://localhost:8765';

export default function App() {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS);
  const { connected, lastMessage, connect, disconnect, send } = useWebSocket(wsUrl);
  const { active, source, startMic, startVideo, stop, frameDataRef } = useAudioStream(send);

  // Pipeline state
  const [probs, setProbs] = useState([0, 0, 0, 0]);
  const [vap, setVap] = useState(null);
  const [stateData, setStateData] = useState(null);
  const [transcript, setTranscript] = useState([]);
  const [faunaMessages, setFaunaMessages] = useState([]);
  const [graphText, setGraphText] = useState(null);
  const [entities, setEntities] = useState([]);
  const [gateOpen, setGateOpen] = useState(false);

  // Supermemory
  const [smGraphData, setSmGraphData] = useState(null);
  const [activeMeeting, setActiveMeeting] = useState('all');

  // Stats
  const [frameCount, setFrameCount] = useState(0);
  const gateTimeout = useRef(null);

  // Process messages from server
  useEffect(() => {
    if (!lastMessage) return;
    const msg = lastMessage;

    if (msg.debug) {
      setFrameCount(msg.debug.frame || 0);
    }
    if (msg.diarization?.latest_probs) {
      setProbs(msg.diarization.latest_probs);
    }
    if (msg.vap) {
      setVap(msg.vap);
    }
    if (msg.state) {
      setStateData(msg.state);
    }
    if (msg.transcript) {
      setTranscript(prev => {
        const next = [...prev, ...msg.transcript];
        return next.length > 500 ? next.slice(-500) : next;
      });
    }
    if (msg.gate_open) {
      const g = msg.gate_open;
      setGateOpen(true);
      clearTimeout(gateTimeout.current);
      gateTimeout.current = setTimeout(() => setGateOpen(false), 2000);
      if (g.decision) {
        setFaunaMessages(prev => [...prev, {
          text: g.decision,
          source: g.source || 'draft',
          timestamp: g.timestamp,
          ai_opening: g.ai_opening,
          mode: g.mode,
        }]);
      }
    }
    if (msg.wake_response) {
      setFaunaMessages(prev => [...prev, {
        text: msg.wake_response.text,
        source: msg.wake_response.source || 'wake',
        timestamp: msg.wake_response.timestamp,
      }]);
    }
    if (msg.graph_text) {
      setGraphText(msg.graph_text);
    }
    if (msg.entities) {
      setEntities(msg.entities);
    }
  }, [lastMessage]);

  const handleConnect = useCallback(() => {
    connect();
  }, [connect]);

  const handleDisconnect = useCallback(() => {
    stop();
    disconnect();
  }, [stop, disconnect]);

  const handleVideoElement = useCallback((el) => {
    startVideo(el);
  }, [startVideo]);

  const contextValue = {
    // Pipeline context
    probs, vap, stateData, transcript, faunaMessages, graphText, entities, gateOpen,
    // Controls context
    active, source, handleVideoElement, frameDataRef,
    // Stats context
    frameCount
  };

  return (
    <BrowserRouter>
      <div className="app-global-controls" style={{
        position: 'absolute', top: '1rem', right: '1rem', zIndex: 100, 
        display: 'flex', gap: '0.5rem', background: 'var(--bg-panel)', padding: '0.5rem', borderRadius: '12px', border: '1px solid var(--border-color)'
      }}>
        <input
          type="text"
          value={wsUrl}
          onChange={(e) => setWsUrl(e.target.value)}
          className="app__ws-input"
          placeholder="ws://..."
          style={{ width: '180px' }}
        />
        {!connected ? (
          <button onClick={handleConnect} className="app__btn app__btn--primary">Connect</button>
        ) : (
          <button onClick={handleDisconnect} className="app__btn app__btn--danger">Disconnect</button>
        )}
        {connected && !active && (
          <button onClick={startMic} className="app__btn app__btn--secondary">🎙 Mic</button>
        )}
        {active && (
          <button onClick={stop} className="app__btn app__btn--ghost">⏹ Stop</button>
        )}
      </div>

      <Routes>
        <Route path="/" element={<Layout connected={connected} apiKey={SM_API_KEY} context={contextValue} />}>
          <Route index element={<Home />} />
          <Route path="intelligence" element={<Intelligence />} />
          <Route 
            path="memory" 
            element={
              <Memory 
                smApiKey={SM_API_KEY} 
                smGraphData={smGraphData} 
                activeMeeting={activeMeeting} 
                setActiveMeeting={setActiveMeeting} 
                setSmGraphData={setSmGraphData} 
              />
            } 
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
