import { NavLink, Outlet } from 'react-router-dom';
import { LayoutGrid, Cpu, BookOpen, Activity } from 'lucide-react';
import './Layout.css';

export default function Layout({ connected, apiKey }) {
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
          <NavLink to="/intelligence" className={({ isActive }) => `layout__link ${isActive ? 'active' : ''}`}>
            <Cpu size={20} />
            Intelligence
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
        <Outlet />
      </main>
    </div>
  );
}
