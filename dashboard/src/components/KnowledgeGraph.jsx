import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import './KnowledgeGraph.css';

const TYPE_COLORS = {
  person: '#60a5fa',
  organization: '#a78bfa',
  product: '#34d399',
  date: '#fbbf24',
  decision: '#f472b6',
  'action item': '#fb923c',
  deadline: '#ef4444',
  topic: '#06b6d4',
  'project name': '#22d3ee',
  default: '#94a3b8',
};

export default function KnowledgeGraph({ graphData, supermemoryData }) {
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const [tab, setTab] = useState('live');

  useEffect(() => {
    if (tab !== 'live' || !graphData) return;
    const svg = d3.select(svgRef.current);
    const width = svgRef.current?.clientWidth || 400;
    const height = svgRef.current?.clientHeight || 300;
    svg.selectAll('*').remove();

    const g = svg.append('g');
    svg.call(d3.zoom().on('zoom', (e) => g.attr('transform', e.transform)));

    const nodes = [];
    const edges = [];
    const nodeMap = {};

    if (typeof graphData === 'string') {
      const lines = graphData.split('\n');
      lines.forEach((line) => {
        const nodeMatch = line.match(/- \[(.+?)\] (.+?) \(x(\d+)\)/);
        if (nodeMatch) {
          const [, type, label, mentions] = nodeMatch;
          const id = `${type}:${label.toLowerCase().trim()}`;
          nodeMap[id] = { id, label, type, mentions: parseInt(mentions) };
          nodes.push(nodeMap[id]);
          return;
        }
        const edgeMatch = line.match(/\s+(.+?) --(.+?)-->\s*(.+)/);
        if (edgeMatch) {
          const [, src, relation, tgt] = edgeMatch;
          edges.push({ source: src.trim(), target: tgt.trim(), relation });
        }
      });
    }

    if (nodes.length === 0) {
      svg.append('text')
        .attr('x', width / 2).attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#64748b').attr('font-size', 12)
        .text('Waiting for entities…');
      return;
    }

    const linkedEdges = edges.map(e => {
      const src = nodes.find(n => n.label === e.source) || nodes[0];
      const tgt = nodes.find(n => n.label === e.target) || nodes[0];
      return { ...e, source: src, target: tgt };
    }).filter(e => e.source !== e.target);

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(linkedEdges).id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => 8 + d.mentions * 3));

    const link = g.append('g')
      .selectAll('line')
      .data(linkedEdges)
      .join('line')
      .attr('stroke', 'rgba(255,255,255,0.08)')
      .attr('stroke-width', 1);

    const nodeG = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    nodeG.append('circle')
      .attr('r', d => 6 + d.mentions * 2)
      .attr('fill', d => TYPE_COLORS[d.type] || TYPE_COLORS.default)
      .attr('opacity', 0.85)
      .attr('stroke', d => TYPE_COLORS[d.type] || TYPE_COLORS.default)
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.3);

    nodeG.append('text')
      .text(d => d.label.length > 18 ? d.label.slice(0, 16) + '…' : d.label)
      .attr('dx', d => 10 + d.mentions * 2)
      .attr('dy', 4)
      .attr('fill', '#c9d1d9')
      .attr('font-size', 10);

    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeG.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    simRef.current = sim;
    return () => sim.stop();
  }, [graphData, tab]);

  useEffect(() => {
    if (tab !== 'supermemory' || !supermemoryData) return;
    const svg = d3.select(svgRef.current);
    const width = svgRef.current?.clientWidth || 400;
    const height = svgRef.current?.clientHeight || 300;
    svg.selectAll('*').remove();

    const { documents, edges: smEdges } = supermemoryData;
    if (!documents || documents.length === 0) {
      svg.append('text')
        .attr('x', width / 2).attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#64748b').attr('font-size', 12)
        .text('No Supermemory data yet');
      return;
    }

    const g = svg.append('g');
    svg.call(d3.zoom().on('zoom', (e) => g.attr('transform', e.transform)));

    const xScale = d3.scaleLinear()
      .domain(d3.extent(documents, d => d.x))
      .range([40, width - 40]);
    const yScale = d3.scaleLinear()
      .domain(d3.extent(documents, d => d.y))
      .range([40, height - 40]);

    if (smEdges) {
      const docMap = {};
      documents.forEach(d => { docMap[d.id] = d; });
      g.append('g').selectAll('line')
        .data(smEdges.filter(e => docMap[e.source] && docMap[e.target]))
        .join('line')
        .attr('x1', e => xScale(docMap[e.source].x))
        .attr('y1', e => yScale(docMap[e.source].y))
        .attr('x2', e => xScale(docMap[e.target].x))
        .attr('y2', e => yScale(docMap[e.target].y))
        .attr('stroke', 'rgba(99,102,241,0.2)')
        .attr('stroke-width', e => 0.5 + (e.similarity || 0) * 2);
    }

    const nodeG = g.append('g').selectAll('g')
      .data(documents)
      .join('g')
      .attr('transform', d => `translate(${xScale(d.x)},${yScale(d.y)})`);

    nodeG.append('circle')
      .attr('r', d => 6 + (d.memories?.length || 0) * 2)
      .attr('fill', 'var(--accent-indigo)')
      .attr('opacity', 0.7)
      .attr('stroke', 'var(--accent-indigo)')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.3);

    nodeG.append('text')
      .text(d => {
        const label = d.title || d.summary || d.id;
        return label.length > 20 ? label.slice(0, 18) + '…' : label;
      })
      .attr('dx', 12).attr('dy', 4)
      .attr('fill', '#c9d1d9')
      .attr('font-size', 10);

    nodeG.append('title')
      .text(d => (d.memories || []).map(m => m.memory).join('\n'));

  }, [supermemoryData, tab]);

  return (
    <div className="kg">
      <div className="kg__tabs">
        <button
          className={`kg__tab ${tab === 'live' ? 'active' : ''}`}
          onClick={() => setTab('live')}
        >
          Live Graph
        </button>
        <button
          className={`kg__tab ${tab === 'supermemory' ? 'active' : ''}`}
          onClick={() => setTab('supermemory')}
        >
          Supermemory
        </button>
      </div>
      <svg ref={svgRef} className="kg__svg" />
      <div className="kg__legend">
        {Object.entries(TYPE_COLORS).filter(([k]) => k !== 'default').map(([type, color]) => (
          <span key={type} className="kg__legend-item">
            <span className="kg__dot" style={{ background: color }} />
            {type}
          </span>
        ))}
      </div>
    </div>
  );
}
