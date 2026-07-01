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

function reachableFromSource(nodes: AppNode[], edges: AppEdge[]): Set<string> {
  const sourceIds = nodes.filter((n) => n.type === 'datasetSource').map((n) => n.id)
  if (sourceIds.length === 0) return new Set<string>()

  const adj = new Map<string, string[]>()
  for (const e of edges) {
    if (!adj.has(e.source)) adj.set(e.source, [])
    adj.get(e.source)!.push(e.target)
  }

  const seen = new Set<string>()
  const queue = [...sourceIds]
  while (queue.length) {
    const id = queue.shift()!
    if (seen.has(id)) continue
    seen.add(id)
    for (const next of adj.get(id) ?? []) queue.push(next)
  }
  return seen
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

// ── Additional transform helpers ──────────────────────────────────────────────

function colMean(X: number[][], j: number): number {
  return X.reduce((s, r) => s + r[j], 0) / X.length
}
function colMedian(X: number[][], j: number): number {
  const sorted = X.map((r) => r[j]).sort((a, b) => a - b)
  const m = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[m - 1] + sorted[m]) / 2 : sorted[m]
}
function colStd(X: number[][], j: number, mean: number): number {
  const variance = X.reduce((s, r) => s + (r[j] - mean) ** 2, 0) / X.length
  return Math.sqrt(variance) || 1
}

// ── Main executor ─────────────────────────────────────────────────────────────

export function executePipeline(
  dataset: LoadedDataset,
  targetColumn: string,
  pipelineNodes: AppNode[],
  pipelineEdges: AppEdge[],
  options?: { maxRows?: number },
): PipelineResult {
  if (!targetColumn) return { ok: false, error: 'No target column selected.' }
  if (!dataset.columns.find((c) => c.name === targetColumn)) {
    return { ok: false, error: `Target column "${targetColumn}" not found in dataset.` }
  }

  const fullRowCount = dataset.rows.length
  const previewRows =
    options?.maxRows != null && fullRowCount > options.maxRows
      ? dataset.rows.slice(0, options.maxRows)
      : dataset.rows
  const scaleToFull = previewRows.length > 0 ? fullRowCount / previewRows.length : 1

  // Default pipeline state
  let rows = [...previewRows]
  let normalize = false
  let normalizeMethod: 'min-max' | 'zscore' = 'min-max'
  let normalizeColumns: string[] = []
  let doShuffle = false
  let shuffleSeed = 42
  let trainRatio = 0.8
  let doOneHot = false
  let oneHotColumns: string[] = []

  // New transform state flags
  let doStandardScaler = false
  let standardScalerColumns: string[] = []
  let doLogTransform = false
  let logTransformColumns: string[] = []
  let doFillNaN = false
  let fillNaNStrategy = 'mean'
  let fillNaNConstant = 0
  let doDropColumns = false
  let dropColumnsList: string[] = []
  let doClipOutliers = false
  let clipStdFactor = 3
  let doBinColumn = false
  let binColumn = ''
  let binCount = 5
  let binStrategy = 'equal-width'
  let doSelectKBest = false
  let selectK = 10
  let doOrdinalEncode = false
  let ordinalEncodeColumns: string[] = []
  let doBalance = false
  let doDropDuplicates = false

  // Walk only nodes reachable from the source (ignore disconnected palette drops)
  const reachable = reachableFromSource(pipelineNodes, pipelineEdges)
  const sorted = topoSort(pipelineNodes, pipelineEdges).filter((n) => reachable.has(n.id))
  for (const node of sorted) {
    const data = node.data as Record<string, unknown>
    switch (node.type) {
      case 'datasetSource':
        break

      case 'normalizeNode':
        normalize = true
        normalizeMethod = (data.method as string) === 'zscore' ? 'zscore' : 'min-max'
        normalizeColumns = data.columns
          ? String(data.columns).split(',').map((s) => s.trim()).filter(Boolean)
          : []
        break

      case 'standardScalerNode':
        doStandardScaler = true
        standardScalerColumns = data.columns
          ? String(data.columns).split(',').map((s) => s.trim()).filter(Boolean)
          : []
        break

      case 'logTransformNode':
        doLogTransform = true
        logTransformColumns = data.columns
          ? String(data.columns).split(',').map((s) => s.trim()).filter(Boolean)
          : []
        break

      case 'clipOutliersNode':
        doClipOutliers = true
        clipStdFactor = typeof data.stdFactor === 'number' ? data.stdFactor : 3
        break

      case 'binColumnNode':
        doBinColumn = true
        binColumn  = String(data.column ?? '')
        binCount   = typeof data.bins === 'number' ? data.bins : 5
        binStrategy = String(data.strategy ?? 'equal-width')
        break

      case 'fillNaNNode':
        doFillNaN = true
        fillNaNStrategy  = String(data.strategy ?? 'mean')
        fillNaNConstant  = typeof data.constant === 'number' ? data.constant : 0
        break

      case 'dropColumnsNode':
        doDropColumns = true
        dropColumnsList = data.columns
          ? String(data.columns).split(',').map((s) => s.trim()).filter(Boolean)
          : []
        break

      case 'dropDuplicatesNode':
        doDropDuplicates = true
        break

      case 'selectKBestNode':
        doSelectKBest = true
        selectK = typeof data.k === 'number' ? data.k : 10
        break

      case 'balanceClassesNode':
        doBalance = true
        break

      case 'oneHotEncodeNode':
        doOneHot = true
        oneHotColumns = data.columns
          ? String(data.columns).split(',').map((s) => s.trim()).filter(Boolean)
          : []
        break

      case 'ordinalEncodeNode':
        doOrdinalEncode = true
        ordinalEncodeColumns = data.columns
          ? String(data.columns).split(',').map((s) => s.trim()).filter(Boolean)
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
        const op  = String(data.operator ?? '>')
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
    }
  }

  // ── Apply row-level transforms ────────────────────────────────────────────

  // Drop duplicates
  if (doDropDuplicates) {
    const seen = new Set<string>()
    rows = rows.filter((row) => {
      const key = JSON.stringify(row)
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
  }

  // Drop named columns
  if (doDropColumns && dropColumnsList.length > 0) {
    rows = rows.map((row) => {
      const out = { ...row }
      for (const col of dropColumnsList) delete out[col]
      return out
    })
  }

  if (rows.length < 4) {
    return { ok: false, error: 'Dataset has too few rows after filtering/deduplication.' }
  }

  // Shuffle
  if (doShuffle) rows = seededShuffle(rows, shuffleSeed)

  // Ordinal encode before one-hot
  if (doOrdinalEncode) {
    const colsToEncode = ordinalEncodeColumns.length > 0
      ? ordinalEncodeColumns
      : dataset.columns.filter((c) => c.type === 'string' && c.name !== targetColumn).map((c) => c.name)
    for (const colName of colsToEncode) {
      const uniq = [...new Set(rows.map((r) => String(r[colName])))].sort()
      const map = new Map(uniq.map((v, i) => [v, i]))
      rows = rows.map((row) => ({ ...row, [colName]: map.get(String(row[colName])) ?? 0 }))
    }
  }

  // One-hot encode categorical feature columns
  if (doOneHot) {
    const colsToEncode = oneHotColumns.length > 0
      ? oneHotColumns
      : dataset.columns.filter((c) => c.type === 'string' && c.name !== targetColumn).map((c) => c.name)
    for (const colName of colsToEncode) {
      const uniq = [...new Set(rows.map((r) => String(r[colName])))]
      const result = expandOneHot(rows, colName, uniq)
      rows = result.newRows
    }
  }

  // Determine feature columns (all remaining numeric cols except target)
  const sampleRow = rows[0]
  let featureNames = Object.keys(sampleRow).filter(
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

  // ── Apply column-level transforms ─────────────────────────────────────────

  // FillNaN (replace NaN / Infinity)
  if (doFillNaN) {
    const fillVals = featureNames.map((_, j) => {
      if (fillNaNStrategy === 'median') return colMedian(X, j)
      if (fillNaNStrategy === 'constant') return fillNaNConstant
      return colMean(X, j) // mean (default)
    })
    X = X.map((row) =>
      row.map((v, j) => (!isFinite(v) || isNaN(v) ? fillVals[j] : v))
    )
  }

  // Log transform (log1p)
  if (doLogTransform) {
    const idxs = logTransformColumns.length > 0
      ? logTransformColumns.map((c) => featureNames.indexOf(c)).filter((i) => i >= 0)
      : featureNames.map((_, i) => i)
    X = X.map((row) => row.map((v, j) => idxs.includes(j) ? Math.log1p(Math.max(0, v)) : v))
  }

  // Clip outliers
  if (doClipOutliers) {
    const stats = featureNames.map((_, j) => {
      const m = colMean(X, j)
      const s = colStd(X, j, m)
      return { lo: m - clipStdFactor * s, hi: m + clipStdFactor * s }
    })
    X = X.map((row) => row.map((v, j) => Math.max(stats[j].lo, Math.min(stats[j].hi, v))))
  }

  // Bin column
  if (doBinColumn && binColumn) {
    const ci = featureNames.indexOf(binColumn)
    if (ci >= 0) {
      const vals = X.map((r) => r[ci])
      const mn = Math.min(...vals), mx = Math.max(...vals)
      if (binStrategy === 'quantile') {
        const sorted = [...vals].sort((a, b) => a - b)
        const q = featureNames.map((_, i) => sorted[Math.floor(i / featureNames.length * sorted.length)] ?? mn)
        X = X.map((row) => {
          const bin = q.findIndex((b) => row[ci] <= b)
          return row.map((v, j) => j === ci ? (bin < 0 ? binCount - 1 : bin) : v)
        })
      } else {
        const step = (mx - mn) / binCount || 1
        X = X.map((row) => row.map((v, j) => j === ci ? Math.min(binCount - 1, Math.floor((v - mn) / step)) : v))
      }
    }
  }

  // Normalise feature matrix (min-max)
  if (normalize) {
    const colsToNorm = normalizeColumns.length > 0
      ? normalizeColumns.map((c) => featureNames.indexOf(c)).filter((i) => i >= 0)
      : null
    if (colsToNorm === null) {
      X = normalizeMethod === 'zscore' ? zscoreNorm(X) : minMaxNorm(X)
    } else {
      const full = normalizeMethod === 'zscore' ? zscoreNorm(X) : minMaxNorm(X)
      X = X.map((row, ri) => row.map((v, ci) => (colsToNorm.includes(ci) ? full[ri][ci] : v)))
    }
  }

  // Standard scaler (z-score per column)
  if (doStandardScaler) {
    const idxs = standardScalerColumns.length > 0
      ? standardScalerColumns.map((c) => featureNames.indexOf(c)).filter((i) => i >= 0)
      : featureNames.map((_, i) => i)
    const stats = idxs.map((j) => { const m = colMean(X, j); return { m, s: colStd(X, j, m) } })
    X = X.map((row) => row.map((v, j) => {
      const pos = idxs.indexOf(j)
      return pos >= 0 ? (v - stats[pos].m) / stats[pos].s : v
    }))
  }

  // Select K best features by variance
  if (doSelectKBest && selectK < featureNames.length) {
    const variances = featureNames.map((_, j) => {
      const m = colMean(X, j)
      return X.reduce((s, r) => s + (r[j] - m) ** 2, 0) / X.length
    })
    const ranked = variances.map((v, i) => ({ i, v })).sort((a, b) => b.v - a.v).slice(0, selectK)
    const keep = new Set(ranked.map((r) => r.i))
    featureNames = featureNames.filter((_, i) => keep.has(i))
    X = X.map((row) => row.filter((_, i) => keep.has(i)))
  }

  // Balance classes (oversample minority)
  if (doBalance) {
    const classCounts = new Map<number, number[]>()
    y.forEach((cls, i) => { if (!classCounts.has(cls)) classCounts.set(cls, []); classCounts.get(cls)!.push(i) })
    const maxCount = Math.max(...[...classCounts.values()].map((v) => v.length))
    const newX: number[][] = [...X]
    const newY: number[] = [...y]
    for (const [cls, idxs] of classCounts) {
      const need = maxCount - idxs.length
      for (let i = 0; i < need; i++) {
        const src = idxs[i % idxs.length]
        newX.push(X[src])
        newY.push(cls)
      }
    }
    X = newX
    // y is already the labels array — overwrite for balance
    y.length = 0
    for (const v of newY) y.push(v)
  }

  // Split
  const splitIdx = Math.floor(X.length * trainRatio)
  const X_train = X.slice(0, splitIdx)
  const y_train = y.slice(0, splitIdx)
  const X_val = X.slice(splitIdx)
  const y_val = y.slice(splitIdx)

  if (X_val.length === 0) {
    return { ok: false, error: 'Validation set is empty. Lower the train ratio or add more data.' }
  }

  const trainSamples =
    scaleToFull > 1 ? Math.round(X_train.length * scaleToFull) : X_train.length
  const valSamples =
    scaleToFull > 1 ? Math.round(X_val.length * scaleToFull) : X_val.length

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
      trainSamples,
      valSamples,
      datasetName: dataset.name,
    },
  }
}
