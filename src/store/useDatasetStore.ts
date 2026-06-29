import { create } from 'zustand'
import Papa from 'papaparse'
import {
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react'
import type { NodeChange, EdgeChange, Connection } from '@xyflow/react'
import type { AppNode, AppEdge } from '../types/graph'

// ── Column metadata ──────────────────────────────────────────────────────────

export type ColumnType = 'number' | 'string' | 'boolean'

export interface ColumnInfo {
  name: string
  type: ColumnType
  nullCount: number
  uniqueCount: number
  min?: number
  max?: number
  mean?: number
  topValues?: string[]
}

// ── Dataset ──────────────────────────────────────────────────────────────────

export interface LoadedDataset {
  id: string
  name: string
  rows: Record<string, unknown>[]
  columns: ColumnInfo[]
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function detectType(values: unknown[]): ColumnType {
  const nonNull = values.filter((v) => v !== null && v !== undefined && v !== '')
  if (nonNull.length === 0) return 'string'
  if (nonNull.every((v) => typeof v === 'boolean')) return 'boolean'
  if (nonNull.every((v) => typeof v === 'number' && !isNaN(v as number))) return 'number'
  return 'string'
}

function computeColumn(name: string, values: unknown[]): ColumnInfo {
  const nonNull = values.filter((v) => v !== null && v !== undefined && v !== '')
  const type = detectType(values)
  const nullCount = values.length - nonNull.length
  const uniqueCount = new Set(nonNull.map(String)).size

  if (type === 'number') {
    const nums = nonNull as number[]
    const min = Math.min(...nums)
    const max = Math.max(...nums)
    const mean = nums.reduce((a, b) => a + b, 0) / nums.length
    return { name, type, nullCount, uniqueCount, min, max, mean }
  }

  if (type === 'string') {
    const freq = new Map<string, number>()
    for (const v of nonNull) {
      const s = String(v)
      freq.set(s, (freq.get(s) ?? 0) + 1)
    }
    const topValues = [...freq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([k]) => k)
    return { name, type, nullCount, uniqueCount, topValues }
  }

  return { name, type, nullCount, uniqueCount }
}

function buildColumns(rows: Record<string, unknown>[]): ColumnInfo[] {
  if (rows.length === 0) return []
  const keys = Object.keys(rows[0])
  return keys.map((key) => computeColumn(key, rows.map((r) => r[key])))
}

function makeSourceNode(datasetName: string): AppNode {
  return {
    id: 'source-1',
    type: 'datasetSource',
    position: { x: 60, y: 120 },
    data: { label: 'Source', datasetName },
  }
}

// ── Store ────────────────────────────────────────────────────────────────────

interface DatasetState {
  dataset: LoadedDataset | null
  targetColumn: string | null
  pipelineNodes: AppNode[]
  pipelineEdges: AppEdge[]

  loadFromCSV: (file: File) => Promise<void>
  loadFromJSON: (file: File) => Promise<void>
  clearDataset: () => void
  setTargetColumn: (col: string | null) => void

  onPipelineNodesChange: (changes: NodeChange<AppNode>[]) => void
  onPipelineEdgesChange: (changes: EdgeChange<AppEdge>[]) => void
  onPipelineConnect: (connection: Connection) => void
  addPipelineNode: (node: AppNode) => void
}

export const useDatasetStore = create<DatasetState>((set, get) => ({
  dataset: null,
  targetColumn: null,
  pipelineNodes: [],
  pipelineEdges: [],

  async loadFromCSV(file) {
    return new Promise((resolve) => {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (result: Papa.ParseResult<Record<string, unknown>>) => {
          const rows = result.data
          const columns = buildColumns(rows)
          const dataset: LoadedDataset = {
            id: `ds-${Date.now()}`,
            name: file.name.replace(/\.csv$/i, ''),
            rows,
            columns,
          }
          const lastCol = columns[columns.length - 1]?.name ?? null
          set({
            dataset,
            targetColumn: lastCol,
            pipelineNodes: [makeSourceNode(dataset.name)],
            pipelineEdges: [],
          })
          resolve()
        },
      })
    })
  },

  async loadFromJSON(file) {
    const text = await file.text()
    try {
      const parsed = JSON.parse(text)
      let rows: Record<string, unknown>[]
      if (Array.isArray(parsed)) {
        rows = parsed as Record<string, unknown>[]
      } else if (typeof parsed === 'object' && parsed !== null) {
        // objects-of-arrays format: { col1: [v1, v2], col2: [v1, v2] }
        const keys = Object.keys(parsed)
        const len = (parsed[keys[0]] as unknown[]).length
        rows = Array.from({ length: len }, (_, i) =>
          Object.fromEntries(keys.map((k) => [k, (parsed[k] as unknown[])[i]]))
        )
      } else {
        return
      }
      const columns = buildColumns(rows)
      const dataset: LoadedDataset = {
        id: `ds-${Date.now()}`,
        name: file.name.replace(/\.json$/i, ''),
        rows,
        columns,
      }
      const lastCol = columns[columns.length - 1]?.name ?? null
      set({
        dataset,
        targetColumn: lastCol,
        pipelineNodes: [makeSourceNode(dataset.name)],
        pipelineEdges: [],
      })
    } catch {
      console.error('Failed to parse JSON dataset')
    }
  },

  clearDataset() {
    set({ dataset: null, targetColumn: null, pipelineNodes: [], pipelineEdges: [] })
  },

  setTargetColumn(col) {
    set({ targetColumn: col })
  },

  onPipelineNodesChange(changes) {
    set({ pipelineNodes: applyNodeChanges(changes, get().pipelineNodes) })
  },

  onPipelineEdgesChange(changes) {
    set({ pipelineEdges: applyEdgeChanges(changes, get().pipelineEdges) })
  },

  onPipelineConnect(connection) {
    set({ pipelineEdges: addEdge({ ...connection, animated: false }, get().pipelineEdges) })
  },

  addPipelineNode(node) {
    set({ pipelineNodes: [...get().pipelineNodes, node] })
  },
}))
