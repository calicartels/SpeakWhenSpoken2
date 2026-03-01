import { useOutletContext } from 'react-router-dom';
import VapGauges from '../components/VapGauges';
import FaunaChat from '../components/FaunaChat';

export default function Intelligence() {
  const {
    vap,
    stateData,
    gateOpen,
    faunaMessages,
    entities
  } = useOutletContext();

  return (
    <div className="page fade-in">
      <header className="page-header">
        <h2 className="page-title">Intelligence</h2>
        <p className="page-subtitle">Real-time LLM decisions, VAP states, and Entity extraction</p>
      </header>

      <div className="app__grid">
        <section className="card card--vap">
          <h2 className="card__title">VAP & Gate Probability</h2>
          <VapGauges vap={vap} state={stateData} gateOpen={gateOpen} />
        </section>

        <section className="card card--fauna">
          <h2 className="card__title">
            Fauna Responses
            {faunaMessages.length > 0 && (
              <span className="card__badge">{faunaMessages.length}</span>
            )}
          </h2>
          <FaunaChat messages={faunaMessages} />
        </section>

        <section className="card card--entities" style={{ gridColumn: '1 / -1' }}>
          <h2 className="card__title">
            Live Entities (GLiNER)
            {entities.length > 0 && (
              <span className="card__badge">{entities.length}</span>
            )}
          </h2>
          <div className="entities-list">
            {entities.length === 0 ? (
              <div className="empty-state">No entities extracted recently...</div>
            ) : (
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', padding: '1rem' }}>
                {entities.map((e, i) => (
                  <span key={i} style={{ 
                    padding: '0.25rem 0.75rem', 
                    background: 'var(--primary-dark)', 
                    borderRadius: '16px',
                    fontSize: '0.85rem',
                    border: '1px solid var(--primary-color)'
                  }}>
                    {e.text} <span style={{ opacity: 0.6, fontSize: '0.75rem', marginLeft: '0.4rem' }}>{e.label}</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
