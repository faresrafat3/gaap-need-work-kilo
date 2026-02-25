'use client';

import { useCallback, useMemo } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

interface ThoughtNode {
  id: string;
  type: 'thought' | 'action' | 'result';
  label: string;
  status: 'active' | 'completed' | 'failed' | 'pending';
  parent?: string;
}

interface ThoughtGraphProps {
  nodes: ThoughtNode[];
  onNodeClick?: (nodeId: string) => void;
}

const statusColors = {
  active: 'bg-layer3 border-layer3',
  completed: 'bg-success/20 border-success',
  failed: 'bg-error/20 border-error',
  pending: 'bg-gray-500/20 border-gray-500',
};

export function ThoughtGraph({ nodes: thoughtNodes, onNodeClick }: ThoughtGraphProps) {
  const initialNodes: Node[] = useMemo(() => {
    return thoughtNodes.map((node, index) => ({
      id: node.id,
      type: 'default',
      data: { 
        label: (
          <div className={`p-2 rounded border-2 ${statusColors[node.status]} min-w-[120px]`}>
            <div className="text-xs font-medium capitalize">{node.type}</div>
            <div className="text-sm mt-1">{node.label}</div>
          </div>
        ),
      },
      position: { x: (index % 3) * 200, y: Math.floor(index / 3) * 100 },
    }));
  }, [thoughtNodes]);

  const initialEdges: Edge[] = useMemo(() => {
    return thoughtNodes
      .filter((node) => node.parent)
      .map((node) => ({
        id: `${node.parent}-${node.id}`,
        source: node.parent!,
        target: node.id,
        animated: node.status === 'active',
      }));
  }, [thoughtNodes]);

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  const handleNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    onNodeClick?.(node.id);
  }, [onNodeClick]);

  return (
    <div className="h-[400px] bg-cyber-dark rounded-lg">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        fitView
        className="bg-cyber-dark"
      >
        <Background color="#4a0e78" gap={16} />
        <Controls className="bg-cyber-darker border border-layer1/30" />
        <MiniMap 
          className="bg-cyber-darker border border-layer1/30"
          nodeColor={(node) => {
            const thoughtNode = thoughtNodes.find(n => n.id === node.id);
            if (!thoughtNode) return '#4a0e78';
            return thoughtNode.status === 'completed' ? '#00c853' :
                   thoughtNode.status === 'failed' ? '#b71c1c' :
                   thoughtNode.status === 'active' ? '#2e7d32' : '#4a0e78';
          }}
        />
      </ReactFlow>
    </div>
  );
}
