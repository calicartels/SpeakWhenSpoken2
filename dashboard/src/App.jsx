import { useState, useEffect, useCallback, useRef } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useWebSocket } from './hooks/useWebSocket';
import { useAudioStream } from './hooks/useAudioStream';
import { useAudioPlayback } from './hooks/useAudioPlayback';

import Layout from './components/Layout';
import Home from './pages/Home';
import Memory from './pages/Memory';

import './App.css';

const SM_API_KEY = import.meta.env.VITE_SUPERMEMORY_API_KEY || '';
const DEFAULT_WS = 'ws://localhost:8765';

export default function App() {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS);
  const { connected, lastMessage, connect, disconnect, send } = useWebSocket(wsUrl);
  const { active, source, startMic, startVideo, stop, frameDataRef } = useAudioStream(send);
  const { play: playTTS } = useAudioPlayback();

  const [probs, setProbs] = useState([0, 0, 0, 0]);
  const [vap, setVap] = useState(null);
  const [stateData, setStateData] = useState(null);
  const [transcript, setTranscript] = useState([]);
  const [faunaMessages, setFaunaMessages] = useState([]);
  const [graphText, setGraphText] = useState(null);
  const [entities, setEntities] = useState([]);
  const [gateOpen, setGateOpen] = useState(false);
  const [liveDraft, setLiveDraft] = useState(null);

  const [smGraphData, setSmGraphData] = useState(null);
  const [activeMeeting, setActiveMeeting] = useState('all');

  const [frameCount, setFrameCount] = useState(0);
  const gateTimeout = useRef(null);

  useEffect(() => {
    if (!lastMessage) return;
    const msg = lastMessage;

    if (msg.debug && msg.debug.frame) {
      setFrameCount((prev) => (prev === msg.debug.frame ? prev : msg.debug.frame));
    }
    if (msg.diarization?.latest_probs) {
      setProbs((prev) => {
        const next = msg.diarization.latest_probs;
        if (!prev || prev.join() !== next.join()) return next;
        return prev;
      });
    }
    if (msg.vap) {
      setVap((prev) => {
        const nextStr = JSON.stringify(msg.vap);
        return prev && JSON.stringify(prev) === nextStr ? prev : msg.vap;
      });
    }
    if (msg.state) {
      setStateData((prev) => (prev === msg.state ? prev : msg.state));
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
      if (g.tts_audio) {
        playTTS(g.tts_audio);
      }
    }
    if (msg.wake_response) {
      setFaunaMessages(prev => [...prev, {
        text: msg.wake_response.text,
        source: msg.wake_response.source || 'wake',
        timestamp: msg.wake_response.timestamp,
      }]);
      if (msg.wake_response.tts_audio) {
        playTTS(msg.wake_response.tts_audio);
      }
    }
    if (msg.graph_text) {
      setGraphText(msg.graph_text);
    }
    if (msg.entities) {
      setEntities(msg.entities);
    }
    if (msg.live_draft) {
      setLiveDraft(msg.live_draft);
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
    probs, vap, stateData, transcript, faunaMessages, graphText, entities, gateOpen,
    active, source, handleVideoElement, frameDataRef,
    frameCount, liveDraft
  };

  return (
    <BrowserRouter>
      <div className="top-global-bar">
        <input
          type="text"
          value={wsUrl}
          onChange={(e) => setWsUrl(e.target.value)}
          className="top-bar__input"
          placeholder="ws://..."
        />
        {!connected ? (
          <button onClick={handleConnect} className="top-bar__btn top-bar__btn--primary">Connect</button>
        ) : (
          <button onClick={handleDisconnect} className="top-bar__btn top-bar__btn--danger">Disconnect</button>
        )}
        {connected && !active && (
          <button onClick={startMic} className="top-bar__btn top-bar__btn--secondary">Mic</button>
        )}
        {active && (
          <button onClick={stop} className="top-bar__btn top-bar__btn--ghost">Stop</button>
        )}
      </div>

      <Routes>
        <Route path="/" element={<Layout connected={connected} apiKey={SM_API_KEY} context={contextValue} />}>
          <Route index element={<Home />} />
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
