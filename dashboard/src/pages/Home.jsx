import { useOutletContext } from 'react-router-dom';
import VideoInput from '../components/VideoInput';
import SpeakerWaveforms from '../components/SpeakerWaveforms';
import Transcript from '../components/Transcript';

export default function Home() {
  const {
    active,
    source,
    probs,
    transcript,
    handleVideoElement,
    frameDataRef
  } = useOutletContext();

  return (
    <div className="page fade-in">
      <header className="page-header">
        <h2 className="page-title">Studio</h2>
        <p className="page-subtitle">Raw audio/video ingestion and live diarization</p>
      </header>

      <div className="app__grid">
        <section className="card card--video">
          <h2 className="card__title">Video / Audio Input</h2>
          <VideoInput onVideoElement={handleVideoElement} active={active && source === 'video'} />
        </section>

        <section className="card card--waveforms">
          <h2 className="card__title">Speaker Waveforms</h2>
          <SpeakerWaveforms probs={probs} frameData={frameDataRef} />
        </section>

        <section className="card card--transcript" style={{ gridColumn: '1 / -1' }}>
          <h2 className="card__title">
            Live Transcript
            {transcript.length > 0 && (
              <span className="card__badge">{transcript.length}</span>
            )}
          </h2>
          <Transcript segments={transcript} />
        </section>
      </div>
    </div>
  );
}
