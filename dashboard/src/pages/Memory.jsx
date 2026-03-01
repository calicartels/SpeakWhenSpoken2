import { useOutletContext } from 'react-router-dom';
import MeetingSidebar from '../components/MeetingSidebar';
import KnowledgeGraph from '../components/KnowledgeGraph';

export default function Memory({ smApiKey, smGraphData, activeMeeting, setActiveMeeting, setSmGraphData }) {
  const { graphText } = useOutletContext();

  return (
    <div className="page fade-in" style={{ display: 'flex', gap: '2rem', height: 'calc(100vh - 8rem)' }}>
      <div style={{ width: '300px', flexShrink: 0 }}>
        <MeetingSidebar
          apiKey={smApiKey}
          activeMeeting={activeMeeting}
          onSelectMeeting={setActiveMeeting}
          onGraphData={setSmGraphData}
        />
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <header className="page-header" style={{ marginBottom: '1rem' }}>
          <h2 className="page-title">Memory Network</h2>
          <p className="page-subtitle">Interactive persistent knowledge graph backed by Supermemory</p>
        </header>

        <section className="card card--graph" style={{ flex: 1, minHeight: 0 }}>
          <KnowledgeGraph graphData={graphText} supermemoryData={smGraphData} />
        </section>
      </div>
    </div>
  );
}
