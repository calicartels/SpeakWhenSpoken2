import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { LayoutGrid, Cpu, BookOpen, Activity } from 'lucide-react';
import VideoInput from './VideoInput';
import './Layout.css';

export default function Layout({ connected, apiKey, context }) {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <div className="layout">
      <aside className="layout__sidebar">
        <div className="layout__brand">
          <span className="layout__logo">🦎</span>
          <h1 className="layout__title">Fauna</h1>
        </div>

        <nav className="layout__nav">
          <NavLink to="/" className={({ isActive }) => `layout__link ${isActive ? 'active' : ''}`}>
            <LayoutGrid size={20} />
            Studio
          </NavLink>
          <NavLink to="/memory" className={({ isActive }) => `layout__link ${isActive ? 'active' : ''}`}>
            <BookOpen size={20} />
            Memory
          </NavLink>
        </nav>

        <div className="layout__footer">
          <div className="layout__status-row">
            <Activity size={16} className={connected ? 'text-green' : 'text-red'} />
            <span className="layout__status-text">
              {connected ? 'Pipeline Live' : 'Offline'}
            </span>
          </div>
          <div className="layout__status-row">
            <span className={`layout__dot ${apiKey ? 'connected' : 'disconnected'}`} />
            <span className="layout__status-text">
              {apiKey ? 'API Active' : 'No API Key'}
            </span>
          </div>
          <div className="layout__version">SpeakWhenSpoken2 v1.1</div>
        </div>
      </aside>

      <main className="layout__main">
        <div style={{ display: isHome ? 'block' : 'none', marginBottom: '1.5rem' }}>
          <section className="card">
            <h2 className="card__title">Video / Audio Input</h2>
            <VideoInput onVideoElement={context.handleVideoElement} active={context.active && context.source === 'video'} />
          </section>
        </div>
        <Outlet context={context} />
      </main>
    </div>
  );
}
