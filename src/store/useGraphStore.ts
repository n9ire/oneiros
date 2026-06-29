import { create } from 'zustand'
import {
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react'
import type {
  NodeChange,
  EdgeChange,
  Connection,
} from '@xyflow/react'
import type { AppNode, AppEdge, NodeData } from '../types/graph'
import type { ValidationIssue } from '../types/validation'
import { validateGraph } from '../editor/validation/validateGraph'

// ── History ──────────────────────────────────────────────────────────────────

interface HistoryEntry {
  nodes: AppNode[]
  edges: AppEdge[]
}

const HISTORY_LIMIT = 60

// ── State shape ──────────────────────────────────────────────────────────────

interface GraphState {
  nodes: AppNode[]
  edges: AppEdge[]
  selectedNodeId: string | null

  // History (undo / redo)
  _past: HistoryEntry[]
  _future: HistoryEntry[]
  canUndo: boolean
  canRedo: boolean
  undo: () => void
  redo: () => void
  _snapshot: () => void

  // Validation
  validationIssues: ValidationIssue[]
  _runValidation: () => void

  // XYFlow handlers
  onNodesChange: (changes: NodeChange<AppNode>[]) => void
  onEdgesChange: (changes: EdgeChange<AppEdge>[]) => void
  onConnect: (connection: Connection) => void

  // Graph mutations
  setSelectedNode: (id: string | null) => void
  addNode: (node: AppNode) => void
  updateNodeData: (id: string, data: Partial<NodeData>) => void

  // Persistence
  loadGraph: (snapshot: { nodes: AppNode[]; edges: AppEdge[] }) => void
  exportGraph: () => { nodes: AppNode[]; edges: AppEdge[] }
}

// ── localStorage helpers ─────────────────────────────────────────────────────

const AUTOSAVE_KEY = 'oneiros-autosave'

function loadPersistedGraph(): { nodes: AppNode[]; edges: AppEdge[] } | null {
  try {
    const raw = localStorage.getItem(AUTOSAVE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as { nodes: AppNode[]; edges: AppEdge[] }
  } catch {
    return null
  }
}

const persisted = loadPersistedGraph()

const defaultNodes: AppNode[] = persisted?.nodes ?? [
  {
    id: 'inputNode-1',
    type: 'inputNode',
    position: { x: 80, y: 200 },
    data: { label: 'Input', batchSize: 1, channels: 1, height: 28, width: 28 },
  },
]
const defaultEdges: AppEdge[] = persisted?.edges ?? []

// ── Store ────────────────────────────────────────────────────────────────────

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: defaultNodes,
  edges: defaultEdges,
  selectedNodeId: null,

  _past: [],
  _future: [],
  canUndo: false,
  canRedo: false,
  validationIssues: validateGraph(defaultNodes, defaultEdges).issues,

  // ── History helpers ──────────────────────────────────────────────────────

  _snapshot() {
    const { nodes, edges, _past } = get()
    const next = [..._past, { nodes, edges }].slice(-HISTORY_LIMIT)
    set({ _past: next, _future: [], canUndo: true, canRedo: false })
  },

  undo() {
    const { _past, _future, nodes, edges } = get()
    if (_past.length === 0) return
    const prev = _past[_past.length - 1]
    const newPast = _past.slice(0, -1)
    const newFuture = [{ nodes, edges }, ..._future]
    set({
      nodes: prev.nodes,
      edges: prev.edges,
      _past: newPast,
      _future: newFuture,
      canUndo: newPast.length > 0,
      canRedo: true,
      selectedNodeId: null,
    })
    get()._runValidation()
  },

  redo() {
    const { _past, _future, nodes, edges } = get()
    if (_future.length === 0) return
    const next = _future[0]
    const newFuture = _future.slice(1)
    const newPast = [..._past, { nodes, edges }]
    set({
      nodes: next.nodes,
      edges: next.edges,
      _past: newPast,
      _future: newFuture,
      canUndo: true,
      canRedo: newFuture.length > 0,
      selectedNodeId: null,
    })
    get()._runValidation()
  },

  // ── Validation ───────────────────────────────────────────────────────────

  _runValidation() {
    const { nodes, edges } = get()
    const { issues } = validateGraph(nodes, edges)
    set({ validationIssues: issues })
  },

  // ── XYFlow handlers ──────────────────────────────────────────────────────

  onNodesChange(changes) {
    const hasDelete = changes.some((c) => c.type === 'remove')
    if (hasDelete) get()._snapshot()
    const updated = applyNodeChanges(changes, get().nodes)
    set({ nodes: updated })
    if (hasDelete) get()._runValidation()
  },

  onEdgesChange(changes) {
    const hasDelete = changes.some((c) => c.type === 'remove')
    if (hasDelete) get()._snapshot()
    const updated = applyEdgeChanges(changes, get().edges)
    set({ edges: updated })
    if (hasDelete) get()._runValidation()
  },

  onConnect(connection) {
    get()._snapshot()
    set({ edges: addEdge({ ...connection, animated: false }, get().edges) })
    get()._runValidation()
  },

  // ── Mutations ────────────────────────────────────────────────────────────

  setSelectedNode(id) {
    set({ selectedNodeId: id })
  },

  addNode(node) {
    get()._snapshot()
    set({ nodes: [...get().nodes, node] })
    get()._runValidation()
  },

  updateNodeData(id, data) {
    get()._snapshot()
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n
      ),
    })
    get()._runValidation()
  },

  // ── Persistence ──────────────────────────────────────────────────────────

  loadGraph({ nodes, edges }) {
    set({
      nodes,
      edges,
      selectedNodeId: null,
      _past: [],
      _future: [],
      canUndo: false,
      canRedo: false,
    })
    get()._runValidation()
  },

  exportGraph() {
    const { nodes, edges } = get()
    return { nodes, edges }
  },
}))
