import { useOutletContext } from 'react-router-dom';
import SpeakerWaveforms from '../components/SpeakerWaveforms';
import Transcript from '../components/Transcript';
import VapGauges from '../components/VapGauges';
import FaunaChat from '../components/FaunaChat';

export default function Home() {
  const {
    probs,
    transcript,
    frameDataRef,
    vap,
    stateData,
    gateOpen,
    faunaMessages,
    liveDraft
  } = useOutletContext();

  return (
    <div className="page fade-in">
      <header className="page-header">
        <h2 className="page-title">Studio</h2>
        <p className="page-subtitle">Raw audio/video ingestion and live diarization</p>
      </header>

      <div className="app__grid">
        <div className="col-span-1" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <section className="card">
            <h2 className="card__title">Speaker Waveforms</h2>
            <SpeakerWaveforms probs={probs} frameData={frameDataRef} />
          </section>

          <section className="card" style={{ flex: 1 }}>
            <h2 className="card__title">
              Live Transcript
              {transcript.length > 0 && (
                <span className="card__badge">{transcript.length}</span>
              )}
            </h2>
            <Transcript segments={transcript} />
          </section>
        </div>

        <div className="col-span-1" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <section className="card">
            <h2 className="card__title">VAP & Gate</h2>
            <VapGauges vap={vap} state={stateData} gateOpen={gateOpen} />
          </section>

          <section className="card" style={{ flex: 1 }}>
            <h2 className="card__title">AI Responses</h2>
            <FaunaChat messages={faunaMessages} liveDraft={liveDraft} />
          </section>
        </div>
      </div>
    </div>
  );
}
