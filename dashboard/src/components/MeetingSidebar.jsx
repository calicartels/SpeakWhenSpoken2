import { useState, useEffect, useCallback } from 'react';
import './MeetingSidebar.css';

const SM_API = 'https://api.supermemory.ai';

export default function MeetingSidebar({ apiKey, activeMeeting, onSelectMeeting, onGraphData }) {
  const [meetings, setMeetings] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchMeetings = useCallback(async () => {
    if (!apiKey) return;
    setLoading(true);
    try {
      const res = await fetch(`${SM_API}/v3/graph/stats`, {
        headers: { Authorization: `Bearer ${apiKey}` },
      });
      if (!res.ok) throw new Error(`Stats HTTP ${res.status}`);
      const data = await res.json();
      setMeetings([{
        id: 'all',
        label: 'All Meetings',
        count: data?.totalDocuments || 0,
        spatial: data?.documentsWithSpatial || 0,
      }]);
    } catch (e) {
      console.warn('Supermemory stats failed:', e.message);
    } finally {
      setLoading(false);
    }
  }, [apiKey]);

  const fetchGraphData = useCallback(async (containerTag) => {
    if (!apiKey) return;
    try {
      const boundsUrl = `${SM_API}/v3/graph/bounds${
        containerTag && containerTag !== 'all' ? `?containerTags=${containerTag}` : ''
      }`;
      const boundsRes = await fetch(boundsUrl, {
        headers: { Authorization: `Bearer ${apiKey}` },
      });
      if (!boundsRes.ok) throw new Error(`Bounds HTTP ${boundsRes.status}`);
      const boundsData = await boundsRes.json();
      const bounds = boundsData?.bounds;
      if (!bounds) return;

      const vpRes = await fetch(`${SM_API}/v3/graph/viewport`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          viewport: bounds,
          ...(containerTag && containerTag !== 'all' ? { containerTags: [containerTag] } : {}),
          limit: 200,
        }),
      });
      if (!vpRes.ok) throw new Error(`Viewport HTTP ${vpRes.status}`);
      const graphData = await vpRes.json();
      onGraphData?.(graphData);
    } catch (e) {
      console.warn('Supermemory graph fetch failed:', e.message);
    }
  }, [apiKey, onGraphData]);

  useEffect(() => {
    fetchMeetings();
    const interval = setInterval(fetchMeetings, 30000);
    return () => clearInterval(interval);
  }, [fetchMeetings]);

  useEffect(() => {
    if (activeMeeting) {
      fetchGraphData(activeMeeting);
      const interval = setInterval(() => fetchGraphData(activeMeeting), 10000);
      return () => clearInterval(interval);
    }
  }, [activeMeeting, fetchGraphData]);

  return (
    <div className="sidebar">
      <div className="sidebar__header">
        <span className="sidebar__logo">🦎</span>
        <h1 className="sidebar__title">Fauna</h1>
      </div>

      <div className="sidebar__section">
        <h2 className="sidebar__section-title">Meetings</h2>
        {loading && <div className="sidebar__loading">Loading…</div>}
        {meetings.map((m) => (
          <button
            key={m.id}
            className={`sidebar__meeting ${activeMeeting === m.id ? 'active' : ''}`}
            onClick={() => onSelectMeeting?.(m.id)}
          >
            <span className="sidebar__meeting-name">{m.label}</span>
            <span className="sidebar__meeting-count">{m.count} docs</span>
          </button>
        ))}
      </div>

      <div className="sidebar__section">
        <h2 className="sidebar__section-title">Connection</h2>
        <div className="sidebar__status-row">
          <span className={`sidebar__dot ${apiKey ? 'connected' : 'disconnected'}`} />
          <span className="sidebar__status-text">
            {apiKey ? 'Supermemory Active' : 'No API Key'}
          </span>
        </div>
      </div>

      <div className="sidebar__footer">
        <span className="sidebar__version">SpeakWhenSpoken2 v1.0</span>
      </div>
    </div>
  );
}
