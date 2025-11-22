// frontend/src/CitationGraph.js
import React, { useEffect, useState } from 'react';
import ReactFlow, { Background, Controls, MiniMap } from 'reactflow';
import 'reactflow/dist/style.css';

export default function CitationGraph({ paperId, onClose }) {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!paperId) return;
    
    setLoading(true);
    fetch(`http://localhost:8000/graph/${paperId}?depth=1`)
      .then(res => res.json())
      .then(data => {
        // Convert to ReactFlow format with circular layout
        const centerNode = data.nodes.find(n => n.isCenter);
        const otherNodes = data.nodes.filter(n => !n.isCenter);
        
        const flowNodes = data.nodes.map((node, i) => {
          let position;
          
          if (node.isCenter) {
            // Center position
            position = { x: 400, y: 300 };
          } else {
            // Circular layout around center
            const angle = (2 * Math.PI * i) / otherNodes.length;
            const radius = 250;
            position = {
              x: 400 + radius * Math.cos(angle),
              y: 300 + radius * Math.sin(angle)
            };
          }
          
          return {
            id: node.id,
            data: { 
              label: (
                <div style={{ 
                  padding: '8px', 
                  textAlign: 'center',
                  maxWidth: '180px' 
                }}>
                  <div style={{ 
                    fontWeight: node.isCenter ? 'bold' : 'normal',
                    fontSize: '11px',
                    marginBottom: '4px'
                  }}>
                    {node.label}
                  </div>
                  <div style={{ 
                    fontSize: '9px', 
                    color: '#666',
                    marginTop: '2px'
                  }}>
                    {node.year} · {node.category}
                  </div>
                </div>
              )
            },
            position,
            style: {
              background: node.isCenter ? '#3182ce' : '#fff',
              color: node.isCenter ? '#fff' : '#000',
              border: `2px solid ${node.isCenter ? '#2c5aa0' : '#cbd5e0'}`,
              borderRadius: '8px',
              padding: '10px',
              width: 200,
              fontSize: '12px',
              boxShadow: node.isCenter 
                ? '0 4px 6px rgba(49, 130, 206, 0.3)'
                : '0 1px 3px rgba(0,0,0,0.1)'
            }
          };
        });
        
        const flowEdges = data.edges.map((edge, i) => ({
          id: `e-${i}`,
          source: edge.source,
          target: edge.target,
          animated: true,
          style: { 
            stroke: '#94a3b8',
            strokeWidth: 2
          }
        }));

        setNodes(flowNodes);
        setEdges(flowEdges);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load graph:", err);
        setLoading(false);
      });
  }, [paperId]);

  if (loading) {
    return (
      <div style={{ 
        padding: '60px', 
        textAlign: 'center',
        background: '#f8fafc'
      }}>
        <div className="spinner"></div>
        <p style={{ marginTop: '20px', color: '#64748b' }}>
          Loading citation network...
        </p>
      </div>
    );
  }

  return (
    <div style={{ height: '600px', background: '#f8fafc' }}>
      <div style={{ 
        padding: '16px 20px',
        background: 'white',
        borderBottom: '1px solid #e2e8f0',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600' }}>
            📊 Citation Network
          </h3>
          <p style={{ 
            margin: '4px 0 0 0', 
            fontSize: '14px', 
            color: '#64748b' 
          }}>
            {nodes.length} papers · {edges.length} connections
          </p>
        </div>
        <button 
          onClick={onClose}
          style={{
            background: '#ef4444',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '6px',
            cursor: 'pointer',
            fontWeight: '500',
            fontSize: '14px'
          }}
        >
          Close
        </button>
      </div>
      
      <div style={{ height: 'calc(100% - 80px)' }}>
        <ReactFlow 
          nodes={nodes} 
          edges={edges}
          fitView
          attributionPosition="bottom-right"
          minZoom={0.5}
          maxZoom={1.5}
        >
          <Background color="#cbd5e0" gap={16} />
          <Controls />
          <MiniMap 
            nodeColor={(node) => node.style.background}
            maskColor="rgba(0, 0, 0, 0.1)"
          />
        </ReactFlow>
      </div>
    </div>
  );
}