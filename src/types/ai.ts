export interface AIMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  actions: AIAction[] | null
  actionsApplied: boolean
  timestamp: number
}

export type AIAction =
  | { type: 'clear' }
  | {
      type: 'addNode'
      id: string
      nodeType: string
      x: number
      y: number
      data: Record<string, unknown>
    }
  | { type: 'connect'; source: string; target: string }

/** Context sent to the backend with each message. */
export interface AIContext {
  graph: { nodes: unknown[]; edges: unknown[] }
  validationErrors: string[]
  datasetInfo: {
    name: string
    rows: number
    features: number
    target: string | null
    classCount: number
  } | null
  trainingResults: {
    epochs: { valAccuracy: number; valLoss: number; trainLoss: number }[]
  } | null
}

/** Parses the ```actions block out of an assistant message. */
export function parseAIMessage(raw: string): { text: string; actions: AIAction[] | null } {
  const match = raw.match(/```actions\s*\n([\s\S]*?)```/)
  if (!match) return { text: raw.trim(), actions: null }
  try {
    const actions = JSON.parse(match[1]) as AIAction[]
    const text = raw.replace(/```actions\s*\n[\s\S]*?```/, '').trim()
    return { text, actions }
  } catch {
    return { text: raw.trim(), actions: null }
  }
}
