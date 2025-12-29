import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import ReactFlow, { Background, Controls, MiniMap, useNodesState, useEdgesState } from 'reactflow';
import 'reactflow/dist/style.css';
import './App.css';
const API_URL = "http://localhost:8000";
function CitationGraph({ paperId, onNodeClick }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  useEffect(() => {
    if(!paperId) return;
    fetch(`${API_URL}/graph/${paperId}`)
      .then(r => r.json())
      .then(data => {
        const unique = new Map();
        data.nodes.forEach(n => unique.set(n.id, n));
        const nodeList = Array.from(unique.values());
        const flowNodes = nodeList.map((n, i) => {
            const isCenter = n.id === paperId;
            const angle = (i / nodeList.length) * 2 * Math.PI;
            const r = isCenter ? 0 : 300;
            return {
                id: n.id,
                position: { x: 400 + r * Math.cos(angle), y: 300 + r * Math.sin(angle) },
                data: { label: <div className="graph-label" style={{color: isCenter?'#000':'#fff', textAlign: 'center', fontSize: '10px'}}>{n.label.slice(0,30)}...</div> },
                style: { 
                    background: isCenter ? '#00ff9d' : '#111', 
                    border: '1px solid #333',
                    width: 160,
                    borderRadius: '8px',
                    color: isCenter ? '#000' : '#ccc',
                    cursor: 'pointer',
                    textAlign: 'center',
                    boxShadow: isCenter ? '0 0 15px rgba(0,255,157,0.4)' : 'none'
                }
            };
        });
        const flowEdges = data.edges.map((e, i) => ({ 
            id: `e${i}`, source: e.source, target: e.target, animated: true, style: { stroke: '#444' } 
        }));
        setNodes(flowNodes);
        setEdges(flowEdges);
      });
  }, [paperId, setNodes, setEdges]);
  const handleNodeClick = useCallback((event, node) => {
      if (onNodeClick) onNodeClick(node.id);
  }, [onNodeClick]);
  return (
    <div style={{width:'100%', height:'100%'}}>
        <ReactFlow nodes={nodes} edges={edges} onNodeClick={handleNodeClick} fitView minZoom={0.1}>
            <Background color="#222" gap={20} />
            <Controls style={{button:{background:'#333', color:'white'}}} />
            <MiniMap nodeColor={n=>n.style.background} maskColor="rgba(0,0,0,0.7)" style={{background:'#000'}} />
        </ReactFlow>
    </div>
  );
}
function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [paper, setPaper] = useState(null);
  const [viewMode, setViewMode] = useState('preview');
  const [chat, setChat] = useState([]);
  const [msg, setMsg] = useState("");
  const [meta, setMeta] = useState(null);
  const search = async () => {
    if (!query) return;
    try {
        const res = await axios.post(`${API_URL}/search`, { query, k: 9 });
        setResults(res.data.results);
        setMeta({ latency: res.data.took_ms, method: res.data.method });
    } catch (e) { alert("Backend Offline"); }
  };
  const openPaper = async (id) => {
    try {
        const res = await axios.get(`${API_URL}/article/${id}`);
        setPaper(res.data);
        setViewMode('preview');
        setChat([{ role: 'ai', text: `System ready. Analyzing "${res.data.title}"...`}]);
    } catch(e) { console.error(e); }
  };
  const sendChat = async () => {
    if (!msg) return;
    const newChat = [...chat, { role: 'user', text: msg }];
    setChat(newChat);
    setMsg("");
    try {
        setChat(prev => [...prev, { role: 'ai', text: "..." }]);
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paper_id: paper.id, message: msg })
        });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let aiText = "";
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            aiText += chunk;
            setChat(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: 'ai', text: aiText };
                return updated;
            });
        }
    } catch (e) { console.error(e); }
  };
  const getAbstract = (p) => p.abstract || p.text || "No abstract available.";
  return (
    <div className="App">
      <div className="header container">
        <div className="brand">MiniVector <span className="badge">V2.0 QUANT</span></div>
        <div style={{color: '#666', fontFamily: 'monospace'}}>SYSTEM: ONLINE</div>
      </div>
      <div className="container">
        <div className="search-box">
            <input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && search()} placeholder="> INPUT SEARCH QUERY" autoFocus />
        </div>
        {meta && (
            <div className="stats-bar">
                <span>FOUND {results.length} RESULTS IN <span style={{color: '#fff'}}>{meta.latency.toFixed(0)}ms</span></span>
                <span className="method-tag">{meta.method}</span>
            </div>
        )}
        <div className="grid">
            {results.map(r => (
                <div key={r.id} className="card" onClick={() => openPaper(r.id)}>
                    <div style={{display:'flex', justifyContent:'space-between', marginBottom:10, fontSize:11, color:'#666', fontFamily: 'monospace'}}>
                        <span>{r.category}</span><span>{(r.score*100).toFixed(1)}% MATCH</span>
                    </div>
                    <h3>{r.title}</h3>
                    <div style={{fontSize:12, color:'#888', marginBottom:8, fontFamily: 'monospace'}}>
                        {Array.isArray(r.authors) ? r.authors.join(", ") : r.authors} • {r.published}
                    </div>
                    <p>{r.text_preview || getAbstract(r).slice(0, 150)}...</p>
                </div>
            ))}
        </div>
      </div>
      {paper && (
        <div className="overlay" onClick={() => setPaper(null)}>
            {viewMode === 'preview' ? (
                <div className="preview-modal" onClick={e => e.stopPropagation()}>
                    <div className="preview-content">
                        <span className="badge">PREVIEW</span>
                        <h1>{paper.title}</h1>
                        <div className="meta-row">
                            {Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors} • {paper.published}
                        </div>
                        <div className="abstract-box">
                            {getAbstract(paper)}
                        </div>
                    </div>
                    <div className="action-bar">
                        {paper.arxiv_id && (
                            <a href={`https: 
                                📄 VIEW PDF
                            </a>
                        )}
                        <button className="btn primary" onClick={() => setViewMode('graph')}>
                            🕸️ VIEW GRAPH
                        </button>
                        <button className="btn text" onClick={() => setPaper(null)}>
                            CLOSE
                        </button>
                    </div>
                </div>
            ) : (
                <div className="cockpit" onClick={e => e.stopPropagation()}>
                    <div className="panel-col">
                        <span className="badge">METADATA</span>
                        <h2 style={{marginTop:15, fontSize: '1.4rem'}}>{paper.title}</h2>
                        <p style={{color:'#888', fontSize:13, fontFamily: 'monospace'}}>
                            {Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors} • {paper.published}
                        </p>
                        <hr style={{borderColor: '#333', margin: '15px 0'}}/>
                        <p style={{lineHeight:1.6, color:'#ccc', fontSize: '0.95rem'}}>{getAbstract(paper)}</p>
                        <button onClick={() => setViewMode('preview')} className="close-btn">BACK TO PREVIEW</button>
                    </div>
                    <div className="graph-col">
                        <div className="graph-header"><span className="badge">CITATION TOPOLOGY</span></div>
                        <CitationGraph paperId={paper.id} onNodeClick={openPaper} />
                    </div>
                    <div className="chat-col">
                        <div className="chat-header">AI ASSISTANT</div>
                        <div className="chat-feed">
                            {chat.map((m, i) => <div key={i} className={`msg ${m.role}`}>{m.text}</div>)}
                        </div>
                        <div className="chat-input-wrapper">
                            <input value={msg} onChange={e => setMsg(e.target.value)} placeholder="Query context..." onKeyDown={e => e.key === 'Enter' && sendChat()} />
                            <button onClick={sendChat}>SEND</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
      )}
    </div>
  );
}
export default App;
