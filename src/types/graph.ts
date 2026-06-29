import type { Node, Edge } from '@xyflow/react'

export interface NodeData extends Record<string, unknown> {
  label: string
}

export type AppNode = Node<NodeData>
export type AppEdge = Edge

export interface GraphSnapshot {
  version: number
  nodes: AppNode[]
  edges: AppEdge[]
}
