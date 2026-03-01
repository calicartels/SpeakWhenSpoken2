import { useState, useRef } from 'react';
import './VideoInput.css';

const YT_REGEX = /(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)([\w-]+)/;
const YT_SERVER = 'http://localhost:3001';

export default function VideoInput({ onVideoElement, active }) {
  const [url, setUrl] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  const isYouTube = (u) => YT_REGEX.test(u);

  const handleLoad = async () => {
    if (!url) return;

    if (isYouTube(url)) {
      setDownloading(true);
      setDownloadStatus('Downloading from YouTube…');
      try {
        const res = await fetch(`${YT_SERVER}/download`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
        const data = await res.json();
        if (data.error) {
          setDownloadStatus(`❌ ${data.error}`);
          setDownloading(false);
          return;
        }
        setDownloadStatus('Loading video…');
        videoRef.current.src = data.videoUrl;
        videoRef.current.load();
      } catch (err) {
        void err;
        setDownloadStatus('❌ Download server not running. Run: npm run yt');
      }
      setDownloading(false);
      return;
    }

    videoRef.current.src = url;
    videoRef.current.load();
  };

  const handleFile = (file) => {
    if (!file) return;
    const objectUrl = URL.createObjectURL(file);
    setUrl(file.name);
    videoRef.current.src = objectUrl;
    videoRef.current.load();
  };

  const handleCanPlay = () => {
    setLoaded(true);
    setDownloadStatus('');
  };

  const handleStreamClick = () => {
    if (videoRef.current && onVideoElement && !streaming) {
      onVideoElement(videoRef.current);
      setStreaming(true);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.type.startsWith('video/') || file.type.startsWith('audio/'))) {
      handleFile(file);
    }
  };

  return (
    <div className="video-input">
      <div className="video-input__controls">
        <input
          type="text"
          placeholder="Paste YouTube URL, video URL, or drag & drop…"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleLoad()}
          className="video-input__url"
        />
        <button
          onClick={handleLoad}
          className="video-input__btn"
          disabled={!url || downloading}
        >
          {downloading ? '⏳' : isYouTube(url) ? '⬇ Download' : 'Load'}
        </button>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="video-input__btn--secondary"
        >
          Browse
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,audio/*"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {downloadStatus && (
        <div className={`video-input__download-status ${downloadStatus.startsWith('❌') ? 'error' : ''}`}>
          {downloadStatus}
        </div>
      )}

      <div
        className={`video-input__player ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <video
          ref={videoRef}
          controls
          onCanPlay={handleCanPlay}
          className="video-input__video"
        />
        {!loaded && (
          <div className="video-input__placeholder">
            <div className="video-input__icon">▶</div>
            <span>Drop a video file or paste a YouTube URL above</span>
          </div>
        )}
      </div>

      {loaded && !streaming && (
        <button onClick={handleStreamClick} className="video-input__stream-btn">
          🔊 Stream Audio to Pipeline
        </button>
      )}

      {active && streaming && (
        <div className="video-input__status">
          <span className="video-input__dot" /> Streaming audio to pipeline in real-time
        </div>
      )}
    </div>
  );
}
