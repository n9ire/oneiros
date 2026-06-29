import { create } from 'zustand'
import { v4 as uuid } from 'uuid'
import type { AIMessage, AIAction, AIContext } from '../types/ai'
import { parseAIMessage } from '../types/ai'
import { useGraphStore } from './useGraphStore'
import { useDatasetStore } from './useDatasetStore'
import { useTrainingStore } from './useTrainingStore'

const API_BASE = 'http://localhost:8000'
const KEY_STORAGE = 'oneiros-openai-key'
const MODEL_STORAGE = 'oneiros-ai-model'

interface AIState {
  messages: AIMessage[]
  apiKey: string
  model: string
  isLoading: boolean

  setApiKey: (key: string) => void
  setModel: (model: string) => void
  sendMessage: (content: string) => Promise<void>
  applyActions: (messageId: string) => void
  clearMessages: () => void
}

function buildContext(): AIContext {
  const graph = useGraphStore.getState().exportGraph()
  const validationIssues = useGraphStore.getState().validationIssues
  const validationErrors = validationIssues.map((i) => `[${i.severity}] ${i.message}`)

  const dsState = useDatasetStore.getState()
  const datasetInfo = dsState.dataset
    ? {
        name: dsState.dataset.name,
        rows: dsState.dataset.rows.length,
        features: dsState.dataset.columns.filter(
          (c) => c.type === 'number' && c.name !== dsState.targetColumn,
        ).length,
        target: dsState.targetColumn,
        classCount: dsState.targetColumn
          ? new Set(dsState.dataset.rows.map((r) => String(r[dsState.targetColumn!]))).size
          : 0,
      }
    : null

  const epochMetrics = useTrainingStore.getState().epochMetrics
  const trainingResults =
    epochMetrics.length > 0
      ? { epochs: epochMetrics.map((e) => ({ valAccuracy: e.valAccuracy, valLoss: e.valLoss, trainLoss: e.trainLoss })) }
      : null

  return { graph, validationErrors, datasetInfo, trainingResults }
}

export const useAIStore = create<AIState>((set, get) => ({
  messages: [],
  apiKey: typeof localStorage !== 'undefined' ? (localStorage.getItem(KEY_STORAGE) ?? '') : '',
  model: typeof localStorage !== 'undefined' ? (localStorage.getItem(MODEL_STORAGE) ?? 'gpt-4o') : 'gpt-4o',
  isLoading: false,

  setApiKey(key) {
    localStorage.setItem(KEY_STORAGE, key)
    set({ apiKey: key })
  },

  setModel(model) {
    localStorage.setItem(MODEL_STORAGE, model)
    set({ model })
  },

  async sendMessage(content) {
    const { apiKey, model, messages } = get()

    const userMsg: AIMessage = {
      id: uuid(),
      role: 'user',
      content,
      actions: null,
      actionsApplied: false,
      timestamp: Date.now(),
    }
    set((s) => ({ messages: [...s.messages, userMsg], isLoading: true }))

    const history = [...messages, userMsg].map((m) => ({ role: m.role, content: m.content }))
    const context = buildContext()

    try {
      const res = await fetch(`${API_BASE}/api/ai/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history, context, apiKey, model }),
      })
      const { message: raw } = (await res.json()) as { message: string }
      const { text, actions } = parseAIMessage(raw)

      const assistantMsg: AIMessage = {
        id: uuid(),
        role: 'assistant',
        content: text,
        actions,
        actionsApplied: false,
        timestamp: Date.now(),
      }
      set((s) => ({ messages: [...s.messages, assistantMsg], isLoading: false }))
    } catch (err) {
      const errorMsg: AIMessage = {
        id: uuid(),
        role: 'assistant',
        content: `Could not reach the backend. Make sure the API server is running on port 8000.\n\n${err instanceof Error ? err.message : String(err)}`,
        actions: null,
        actionsApplied: false,
        timestamp: Date.now(),
      }
      set((s) => ({ messages: [...s.messages, errorMsg], isLoading: false }))
    }
  },

  applyActions(messageId) {
    const msg = get().messages.find((m) => m.id === messageId)
    if (!msg?.actions) return

    applyGraphActions(msg.actions)

    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === messageId ? { ...m, actionsApplied: true } : m,
      ),
    }))
  },

  clearMessages() {
    set({ messages: [] })
  },
}))

// ── Graph action executor ─────────────────────────────────────────────────────

function applyGraphActions(actions: AIAction[]) {
  const store = useGraphStore.getState()
  store._snapshot()

  let nodes = [...store.nodes]
  let edges = [...store.edges]

  for (const action of actions) {
    if (action.type === 'clear') {
      nodes = []
      edges = []
    } else if (action.type === 'addNode') {
      const existing = nodes.findIndex((n) => n.id === action.id)
      const newNode = {
        id: action.id,
        type: action.nodeType,
        position: { x: action.x, y: action.y },
        data: { label: action.nodeType, ...action.data },
      }
      if (existing >= 0) {
        nodes[existing] = newNode
      } else {
        nodes.push(newNode)
      }
    } else if (action.type === 'connect') {
      const edgeId = `ai-${action.source}-${action.target}`
      if (!edges.find((e) => e.id === edgeId)) {
        edges.push({
          id: edgeId,
          source: action.source,
          target: action.target,
          animated: false,
        })
      }
    }
  }

  store.loadGraph({ nodes, edges })
}
