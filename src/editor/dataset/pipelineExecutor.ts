/**
 * Executes the dataset preprocessing pipeline entirely in the browser.
 * Walks pipeline nodes in topological order, applies each transformation,
 * then returns train/val arrays ready to POST to the backend.
 */

import type { AppNode, AppEdge } from '../../types/graph'
import type { LoadedDataset } from '../../store/useDatasetStore'

// ── Output type ───────────────────────────────────────────────────────────────

export interface ProcessedDataset {
  X_train: number[][]
  y_train: number[]
  X_val: number[][]
  y_val: number[]
  featureNames: string[]
  featureCount: number
  classNames: string[]
  classCount: number
  trainSamples: number
  valSamples: number
  datasetName: string
}

export type PipelineResult = { ok: true; data: ProcessedDataset } | { ok: false; error: string }

// ── Topological sort ──────────────────────────────────────────────────────────

function topoSort(nodes: AppNode[], edges: AppEdge[]): AppNode[] {
  const inDegree = new Map<string, number>(nodes.map((n) => [n.id, 0]))
  const adj = new Map<string, string[]>(nodes.map((n) => [n.id, []]))
  for (const e of edges) {
    adj.get(e.source)?.push(e.target)
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1)
  }
  const queue = nodes.filter((n) => inDegree.get(n.id) === 0)
  const result: AppNode[] = []
  while (queue.length) {
    const node = queue.shift()!
    result.push(node)
    for (const nid of adj.get(node.id) ?? []) {
      const d = (inDegree.get(nid) ?? 0) - 1
      inDegree.set(nid, d)
      if (d === 0) queue.push(nodes.find((n) => n.id === nid)!)
    }
  }
  return result.filter(Boolean)
}

// ── Transformations ───────────────────────────────────────────────────────────

function seededShuffle<T>(arr: T[], seed: number): T[] {
  const out = [...arr]
  let s = seed
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return (s >>> 0) / 0xffffffff
  }
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

function minMaxNorm(X: number[][]): number[][] {
  if (X.length === 0) return X
  const nf = X[0].length
  const mins = Array(nf).fill(Infinity)
  const maxs = Array(nf).fill(-Infinity)
  for (const row of X) {
    for (let j = 0; j < nf; j++) {
      if (row[j] < mins[j]) mins[j] = row[j]
      if (row[j] > maxs[j]) maxs[j] = row[j]
    }
  }
  return X.map((row) =>
    row.map((v, j) => {
      const r = maxs[j] - mins[j]
      return r === 0 ? 0 : (v - mins[j]) / r
    }),
  )
}

function zscoreNorm(X: number[][]): number[][] {
  if (X.length === 0) return X
  const nf = X[0].length
  const means = Array(nf).fill(0)
  for (const row of X) {
    for (let j = 0; j < nf; j++) means[j] += row[j]
  }
  for (let j = 0; j < nf; j++) means[j] /= X.length
  const stds = Array(nf).fill(0)
  for (const row of X) {
    for (let j = 0; j < nf; j++) stds[j] += (row[j] - means[j]) ** 2
  }
  for (let j = 0; j < nf; j++) stds[j] = Math.sqrt(stds[j] / X.length) || 1
  return X.map((row) => row.map((v, j) => (v - means[j]) / stds[j]))
}

function encodeLabels(values: unknown[]): { encoded: number[]; classes: string[] } {
  const classes = [...new Set(values.map(String))].sort()
  const classMap = new Map(classes.map((c, i) => [c, i]))
  return { encoded: values.map((v) => classMap.get(String(v)) ?? 0), classes }
}

function expandOneHot(
  rows: Record<string, unknown>[],
  columnName: string,
  classes: string[],
): { newRows: Record<string, unknown>[]; newColNames: string[] } {
  const newColNames = classes.map((c) => `${columnName}_${c}`)
  const newRows = rows.map((row) => {
    const out = { ...row }
    for (const c of classes) {
      out[`${columnName}_${c}`] = String(row[columnName]) === c ? 1 : 0
    }
    delete out[columnName]
    return out
  })
  return { newRows, newColNames }
}

// ── Main executor ─────────────────────────────────────────────────────────────

export function executePipeline(
  dataset: LoadedDataset,
  targetColumn: string,
  pipelineNodes: AppNode[],
  pipelineEdges: AppEdge[],
): PipelineResult {
  if (!targetColumn) return { ok: false, error: 'No target column selected.' }
  if (!dataset.columns.find((c) => c.name === targetColumn)) {
    return { ok: false, error: `Target column "${targetColumn}" not found in dataset.` }
  }

  // Default pipeline state
  let rows = [...dataset.rows]
  let normalize = false
  let normalizeMethod: 'min-max' | 'zscore' = 'min-max'
  let normalizeColumns: string[] = []
  let doShuffle = false
  let shuffleSeed = 42
  let trainRatio = 0.8
  let doOneHot = false
  let oneHotColumns: string[] = []

  // Walk pipeline nodes in topological order
  const sorted = topoSort(pipelineNodes, pipelineEdges)
  for (const node of sorted) {
    const data = node.data as Record<string, unknown>
    switch (node.type) {
      case 'datasetSource':
        break
      case 'normalizeNode':
        normalize = true
        normalizeMethod = (data.method as string) === 'zscore' ? 'zscore' : 'min-max'
        normalizeColumns = data.columns
          ? String(data.columns)
              .split(',')
              .map((s) => s.trim())
              .filter(Boolean)
          : []
        break
      case 'shuffleNode':
        doShuffle = true
        shuffleSeed = typeof data.seed === 'number' ? data.seed : 42
        break
      case 'splitNode':
        trainRatio = typeof data.trainRatio === 'number' ? data.trainRatio : 0.8
        break
      case 'filterNode': {
        const col = String(data.column ?? '')
        const op = String(data.operator ?? '>')
        const val = parseFloat(String(data.value ?? '0'))
        if (col) {
          rows = rows.filter((row) => {
            const v = Number(row[col])
            if (op === '>') return v > val
            if (op === '<') return v < val
            if (op === '>=') return v >= val
            if (op === '<=') return v <= val
            if (op === '==') return v === val
            if (op === '!=') return v !== val
            return true
          })
        }
        break
      }
      case 'oneHotEncodeNode':
        doOneHot = true
        oneHotColumns = data.columns
          ? String(data.columns)
              .split(',')
              .map((s) => s.trim())
              .filter(Boolean)
          : []
        break
    }
  }

  if (rows.length < 4) {
    return { ok: false, error: 'Dataset has too few rows after filtering.' }
  }

  // Shuffle
  if (doShuffle) rows = seededShuffle(rows, shuffleSeed)

  // One-hot encode categorical feature columns
  if (doOneHot) {
    const colsToEncode =
      oneHotColumns.length > 0
        ? oneHotColumns
        : dataset.columns
            .filter((c) => c.type === 'string' && c.name !== targetColumn)
            .map((c) => c.name)

    for (const colName of colsToEncode) {
      const uniq = [...new Set(rows.map((r) => String(r[colName])))]
      const result = expandOneHot(rows, colName, uniq)
      rows = result.newRows
    }
  }

  // Determine feature columns (all remaining numeric cols except target)
  const sampleRow = rows[0]
  const featureNames = Object.keys(sampleRow).filter(
    (k) => k !== targetColumn && typeof sampleRow[k] === 'number',
  )

  if (featureNames.length === 0) {
    return {
      ok: false,
      error: 'No numeric feature columns found. Make sure features are numeric (not the target column).',
    }
  }

  // Extract X (numeric features only)
  let X = rows.map((row) => featureNames.map((f) => Number(row[f] ?? 0)))

  // Extract y (target)
  const rawTargets = rows.map((r) => r[targetColumn])
  const { encoded: y, classes: classNames } = encodeLabels(rawTargets)

  // Normalise feature matrix
  if (normalize) {
    const colsToNorm =
      normalizeColumns.length > 0
        ? normalizeColumns.map((c) => featureNames.indexOf(c)).filter((i) => i >= 0)
        : null // null = all

    if (colsToNorm === null) {
      X = normalizeMethod === 'zscore' ? zscoreNorm(X) : minMaxNorm(X)
    } else {
      // Partial normalisation: only apply to selected indices
      const full = normalizeMethod === 'zscore' ? zscoreNorm(X) : minMaxNorm(X)
      X = X.map((row, ri) =>
        row.map((v, ci) => (colsToNorm.includes(ci) ? full[ri][ci] : v)),
      )
    }
  }

  // Split
  const splitIdx = Math.floor(rows.length * trainRatio)
  const X_train = X.slice(0, splitIdx)
  const y_train = y.slice(0, splitIdx)
  const X_val = X.slice(splitIdx)
  const y_val = y.slice(splitIdx)

  if (X_val.length === 0) {
    return { ok: false, error: 'Validation set is empty. Lower the train ratio or add more data.' }
  }

  return {
    ok: true,
    data: {
      X_train,
      y_train,
      X_val,
      y_val,
      featureNames,
      featureCount: featureNames.length,
      classNames,
      classCount: classNames.length,
      trainSamples: X_train.length,
      valSamples: X_val.length,
      datasetName: dataset.name,
    },
  }
}
