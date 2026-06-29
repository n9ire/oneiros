import { useCallback, useRef } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  useReactFlow,
} from '@xyflow/react'
import type { DragEvent } from 'react'
import '@xyflow/react/dist/style.css'

import './nodes/index'
import { getXYFlowNodeTypes, getNodeDefinition } from './registry/nodeRegistry'
import { useGraphStore } from '../store/useGraphStore'

const nodeTypes = getXYFlowNodeTypes()

function FlowEditorInner() {
  const containerRef = useRef<HTMLDivElement>(null)
  const { screenToFlowPosition } = useReactFlow()

  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const onNodesChange = useGraphStore((s) => s.onNodesChange)
  const onEdgesChange = useGraphStore((s) => s.onEdgesChange)
  const onConnect = useGraphStore((s) => s.onConnect)
  const setSelectedNode = useGraphStore((s) => s.setSelectedNode)
  const addNode = useGraphStore((s) => s.addNode)

  const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      const type = e.dataTransfer.getData('application/oneiros-node')
      if (!type) return

      const def = getNodeDefinition(type)
      if (!def) return

      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })

      addNode({
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: { ...def.defaultData },
      })
    },
    [screenToFlowPosition, addNode]
  )

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={(_, node) => setSelectedNode(node.id)}
        onPaneClick={() => setSelectedNode(null)}
        onDragOver={onDragOver}
        onDrop={onDrop}
        defaultEdgeOptions={{
          style: { stroke: '#52525b', strokeWidth: 1.5 },
          animated: false,
        }}
        proOptions={{ hideAttribution: true }}
        minZoom={0.2}
        maxZoom={3}
        fitView
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#27272a"
        />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor={(n) => {
            const typeColors: Record<string, string> = {
              inputNode: '#3b82f6',
              denseNode: '#8b5cf6',
              outputNode: '#10b981',
            }
            return typeColors[n.type ?? ''] ?? '#3f3f46'
          }}
          maskColor="rgba(9,9,11,0.7)"
          style={{ background: '#18181b' }}
        />
      </ReactFlow>
    </div>
  )
}

export default function FlowEditor() {
  return (
    <ReactFlowProvider>
      <FlowEditorInner />
    </ReactFlowProvider>
  )
}
