import { create } from 'zustand'
import type {
  TrainingStatus,
  TrainingConfig,
  EpochMetrics,
  TrainingMessage,
  CustomDatasetPayload,
} from '../types/training'
import { DEFAULT_CONFIG } from '../types/training'
import { TrainingSocket } from '../services/trainingSocket'
import { useGraphStore } from './useGraphStore'
import { useDatasetStore } from './useDatasetStore'
import { executePipeline } from '../editor/dataset/pipelineExecutor'

const API_BASE = 'http://localhost:8000'

interface TrainingState {
  status: TrainingStatus
  statusMessage: string
  config: TrainingConfig
  runId: string | null

  epochMetrics: EpochMetrics[]
  currentEpoch: number
  totalEpochs: number
  currentBatch: number
  totalBatches: number
  currentLoss: number | null
  etaSecs: number | null
  errorMessage: string | null

  // Custom dataset info (set when using CSV data)
  customDatasetInfo: CustomDatasetPayload | null

  setConfig: (patch: Partial<TrainingConfig>) => void
  startTraining: () => Promise<void>
  stopTraining: () => Promise<void>

  // Model export
  exportWeights: () => void
  exportONNX: () => void
  exportFull: () => void

  _socket: TrainingSocket | null
  _handleMessage: (msg: TrainingMessage) => void
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  status: 'idle',
  statusMessage: '',
  config: { ...DEFAULT_CONFIG },
  runId: null,

  epochMetrics: [],
  currentEpoch: 0,
  totalEpochs: 0,
  currentBatch: 0,
  totalBatches: 0,
  currentLoss: null,
  etaSecs: null,
  errorMessage: null,
  customDatasetInfo: null,

  _socket: null,

  setConfig(patch) {
    set((s) => ({ config: { ...s.config, ...patch } }))
  },

  async startTraining() {
    const { config, _handleMessage } = get()
    const graph = useGraphStore.getState().exportGraph()

    // Build custom dataset payload if CSV mode is selected
    let customDataset: CustomDatasetPayload | null = null
    if (config.dataset === 'custom') {
      const dsState = useDatasetStore.getState()
      if (!dsState.dataset || !dsState.targetColumn) {
        set({ status: 'error', errorMessage: 'Load a dataset and select a target column first.' })
        return
      }
      const result = executePipeline(
        dsState.dataset,
        dsState.targetColumn,
        dsState.pipelineNodes,
        dsState.pipelineEdges,
      )
      if (!result.ok) {
        set({ status: 'error', errorMessage: result.error })
        return
      }
      customDataset = result.data
    }

    set({
      status: 'connecting',
      statusMessage: 'Connecting to backend…',
      epochMetrics: [],
      currentEpoch: 0,
      totalBatches: 0,
      currentBatch: 0,
      currentLoss: null,
      etaSecs: null,
      errorMessage: null,
      customDatasetInfo: customDataset,
    })

    try {
      const res = await fetch(`${API_BASE}/api/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph, config, customDataset }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const { runId } = await res.json() as { runId: string }

      set({ runId })

      const socket = new TrainingSocket()
      set({ _socket: socket })
      socket.connect(`${API_BASE.replace('http', 'ws')}/ws/training/${runId}`, _handleMessage)

    } catch (err) {
      set({
        status: 'error',
        errorMessage: err instanceof Error ? err.message : 'Connection failed',
        statusMessage: '',
      })
    }
  },

  exportWeights() {
    const { runId } = get()
    if (!runId) return
    window.open(`${API_BASE}/api/export/${runId}/weights?filename=model.pt`, '_blank')
  },

  exportONNX() {
    const { runId } = get()
    if (!runId) return
    window.open(`${API_BASE}/api/export/${runId}/onnx?filename=model.onnx`, '_blank')
  },

  exportFull() {
    const { runId } = get()
    if (!runId) return
    window.open(`${API_BASE}/api/export/${runId}/full?filename=model_full.pt`, '_blank')
  },

  async stopTraining() {
    const { runId } = get()
    if (!runId) return
    try {
      await fetch(`${API_BASE}/api/train/stop/${runId}`, { method: 'POST' })
    } catch {
      // ignore network error — the stop event will fire via WebSocket
    }
  },

  _handleMessage(msg) {
    const type = msg.type as string

    if (type === 'status') {
      set({ status: 'running', statusMessage: msg.message as string })
    } else if (type === 'epochStart') {
      set({
        status: 'running',
        currentEpoch: msg.epoch as number,
        totalEpochs: msg.totalEpochs as number,
        currentBatch: 0,
        statusMessage: `Epoch ${msg.epoch} / ${msg.totalEpochs}`,
      })
    } else if (type === 'batch') {
      set({
        currentBatch: msg.batch as number,
        totalBatches: msg.totalBatches as number,
        currentLoss: msg.loss as number,
      })
    } else if (type === 'epochEnd') {
      const entry: EpochMetrics = {
        epoch: msg.epoch as number,
        trainLoss: msg.trainLoss as number,
        valLoss: msg.valLoss as number,
        valAccuracy: msg.valAccuracy as number,
        currentLR: msg.currentLR as number | undefined,
      }
      set((s) => ({
        epochMetrics: [...s.epochMetrics, entry],
        etaSecs: msg.etaSecs as number,
        statusMessage: `Epoch ${msg.epoch} / ${msg.totalEpochs} — val acc ${((msg.valAccuracy as number) * 100).toFixed(1)}%`,
      }))
    } else if (type === 'complete') {
      get()._socket?.close()
      set({ status: 'complete', statusMessage: 'Training complete', _socket: null })
    } else if (type === 'stopped') {
      get()._socket?.close()
      set({ status: 'stopped', statusMessage: 'Training stopped', _socket: null })
    } else if (type === 'error') {
      get()._socket?.close()
      set({ status: 'error', errorMessage: msg.message as string, _socket: null })
    }
  },
}))
