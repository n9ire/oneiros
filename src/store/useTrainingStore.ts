import { create } from 'zustand'
import type {
  TrainingStatus,
  TrainingConfig,
  EpochMetrics,
  TrainingMessage,
  CustomDatasetPayload,
} from '../types/training'
import type { ValidationIssue } from '../types/validation'
import { DEFAULT_CONFIG } from '../types/training'
import { TrainingSocket } from '../services/trainingSocket'
import { useGraphStore } from './useGraphStore'
import { useDatasetStore } from './useDatasetStore'
import { executePipeline } from '../editor/dataset/pipelineExecutor'
import { validateGraph } from '../editor/validation/validateGraph'

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

  preflightIssues: ValidationIssue[]

  setConfig: (patch: Partial<TrainingConfig>) => void
  startTraining: () => Promise<void>
  stopTraining: () => Promise<void>

  // Model export (NN)
  exportWeights: () => void
  exportONNX: () => void
  exportFull: () => void

  // EDF export
  setCustomDatasetFromEDF: (payload: CustomDatasetPayload) => void

  // CV
  cvDatasetRef: { sessionId: string; augmentSteps: unknown[] } | null
  setCVDataset: (ref: { sessionId: string; augmentSteps: unknown[] }) => void
  confusionMatrix: number[][] | null

  // XGBoost
  xgbStatus: 'idle' | 'running' | 'complete' | 'error'
  xgbResult: XGBResult | null
  xgbError: string | null
  trainXGBoost: () => Promise<void>
  exportXGB: () => void

  _socket: TrainingSocket | null
  _handleMessage: (msg: TrainingMessage) => void
}

export interface XGBResult {
  task: 'classification' | 'regression'
  // Classification
  trainAccuracy?: number
  valAccuracy?: number
  nClasses?: number
  // Regression
  trainRMSE?: number
  valRMSE?: number
  trainMAE?: number
  valMAE?: number
  valR2?: number
  objective?: string
  // Common
  bestIteration: number
  nEstimators: number
  featureImportance: { name: string; importance: number }[]
  evals: { round: number; trainLoss: number; valLoss: number }[]
  runId: string
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
  cvDatasetRef: null,
  confusionMatrix: null,

  preflightIssues: [],

  _socket: null,

  setCVDataset(ref) {
    set({ cvDatasetRef: ref })
  },

  setCustomDatasetFromEDF(payload) {
    set({ customDatasetInfo: payload })
    // Auto-select custom dataset mode so training uses this data
    const { config } = get()
    if (config.dataset !== 'custom') {
      set({ config: { ...config, dataset: 'custom' } })
    }
  },

  xgbStatus: 'idle',
  xgbResult: null,
  xgbError: null,

  setConfig(patch) {
    set((s) => ({ config: { ...s.config, ...patch } }))
  },

  async startTraining() {
    const { config, _handleMessage } = get()
    const { nodes, edges } = useGraphStore.getState().exportGraph()

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

    // For image_folder mode, grab the cvDatasetRef from store
    const cvDataset = config.dataset === 'image_folder' ? get().cvDatasetRef : null
    if (config.dataset === 'image_folder' && !cvDataset) {
      set({ status: 'error', errorMessage: 'No image dataset loaded. Import a zip of class folders in the Dataset tab first.' })
      return
    }

    // Pre-flight graph validation
    const dsForValidation = useDatasetStore.getState().cvDataset
    const opts = config.dataset === 'custom' && customDataset
      ? { customFeatureCount: customDataset.featureCount, customClassCount: customDataset.classCount }
      : config.dataset === 'image_folder' && dsForValidation
      ? { cvInputShape: dsForValidation.inputShape, cvClassCount: dsForValidation.classNames.length }
      : {}
    const { issues, isValid } = validateGraph(nodes, edges, opts)
    set({ preflightIssues: issues })
    if (!isValid) {
      const errorCount = issues.filter(i => i.severity === 'error').length
      set({ status: 'error', errorMessage: `Fix ${errorCount} error${errorCount > 1 ? 's' : ''} in your graph before training.` })
      return
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
        body: JSON.stringify({ graph: { nodes, edges }, config, customDataset, cvDataset }),
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

  async trainXGBoost() {
    const { config } = get()
    const dsState = useDatasetStore.getState()

    if (!dsState.dataset || !dsState.targetColumn) {
      set({ xgbStatus: 'error', xgbError: 'Load a CSV dataset and set a target column first.' })
      return
    }

    const result = executePipeline(
      dsState.dataset,
      dsState.targetColumn,
      dsState.pipelineNodes,
      dsState.pipelineEdges,
    )
    if (!result.ok) {
      set({ xgbStatus: 'error', xgbError: result.error })
      return
    }

    set({ xgbStatus: 'running', xgbError: null, xgbResult: null })

    try {
      const res = await fetch(`${API_BASE}/api/xgboost/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: result.data,
          config: {
            task:                 config.xgbTask,
            objective:            config.xgbObjective || undefined,
            nEstimators:          config.xgbNEstimators,
            maxDepth:             config.xgbMaxDepth,
            learningRate:         config.xgbLearningRate,
            subsample:            config.xgbSubsample,
            colsampleBytree:      config.xgbColsampleBytree,
            minChildWeight:       config.xgbMinChildWeight,
            gamma:                config.xgbGamma,
            regAlpha:             config.xgbRegAlpha,
            regLambda:            config.xgbRegLambda,
            earlyStoppingRounds:  config.xgbEarlyStoppingRounds,
          },
        }),
      })

      const json = await res.json()
      if (!res.ok) {
        set({ xgbStatus: 'error', xgbError: json.error ?? `HTTP ${res.status}` })
        return
      }

      set({ xgbStatus: 'complete', xgbResult: json as import('../store/useTrainingStore').XGBResult })
    } catch (err) {
      set({ xgbStatus: 'error', xgbError: err instanceof Error ? err.message : 'Request failed' })
    }
  },

  exportXGB() {
    const { xgbResult } = get()
    if (!xgbResult?.runId) return
    window.open(`${API_BASE}/api/xgboost/${xgbResult.runId}/export`, '_blank')
  },

  async stopTraining() {
    const { runId, _socket } = get()

    // Always close the socket immediately so the UI transitions out of running
    _socket?.close()
    set({ status: 'stopped', statusMessage: 'Training stopped', _socket: null })

    if (!runId) {
      // Clicked Stop before runId was returned — nothing more to do
      return
    }
    try {
      await fetch(`${API_BASE}/api/train/stop/${runId}`, { method: 'POST' })
    } catch {
      // Network error is fine — the stop event fires on WebSocketDisconnect too
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
        top5Accuracy: msg.top5Accuracy as number | undefined,
        currentLR: msg.currentLR as number | undefined,
      }
      const accStr = `${((msg.valAccuracy as number) * 100).toFixed(1)}%`
      const top5Str = entry.top5Accuracy != null ? ` · top-5 ${(entry.top5Accuracy * 100).toFixed(1)}%` : ''
      set((s) => ({
        epochMetrics: [...s.epochMetrics, entry],
        etaSecs: msg.etaSecs as number,
        statusMessage: `Epoch ${msg.epoch} / ${msg.totalEpochs} — val acc ${accStr}${top5Str}`,
      }))
    } else if (type === 'warning') {
      set({ statusMessage: `⚠ ${msg.message as string}` })
    } else if (type === 'complete') {
      get()._socket?.close()
      const cm = (msg as Record<string, unknown>).confusionMatrix
      set({ status: 'complete', statusMessage: 'Training complete', _socket: null, confusionMatrix: cm as number[][] | null ?? null })
    } else if (type === 'stopped') {
      get()._socket?.close()
      set({ status: 'stopped', statusMessage: 'Training stopped', _socket: null })
    } else if (type === 'error') {
      get()._socket?.close()
      set({ status: 'error', errorMessage: msg.message as string, _socket: null })
    }
  },
}))
