import type { LoadedDataset } from '../../store/useDatasetStore'
import type { AppNode, AppEdge } from '../../types/graph'
import { executePipeline } from './pipelineExecutor'

export interface TabularModelInfo {
  totalRows: number
  previewRows: number | null
  numericFeatures: number
  categoricalFeatures: number
  featureCount: number
  targetColumn: string | null
  taskType: 'classification' | 'regression' | 'unset'
  classCount: number | null
  classPreview: string | null
  inputShape: string | null
  outputClasses: number | null
  trainSamples: number | null
  valSamples: number | null
  source: 'raw' | 'pipeline'
  pipelineError: string | null
  missingTargets: number
}

export function inferTaskType(
  targetCol: LoadedDataset['columns'][number] | undefined,
  uniqueCount: number,
): 'classification' | 'regression' {
  if (!targetCol) return 'classification'
  if (targetCol.type === 'string' || targetCol.type === 'boolean') return 'classification'
  if (targetCol.type === 'number') return uniqueCount <= 50 ? 'classification' : 'regression'
  return 'classification'
}

function targetStats(dataset: LoadedDataset, targetColumn: string) {
  const values = dataset.rows
    .map((r) => r[targetColumn])
    .filter((v) => v !== null && v !== undefined && v !== '')
  const missing = dataset.rows.length - values.length
  const unique = [...new Set(values.map(String))].sort()
  return { missing, unique, count: unique.length }
}

function formatClassPreview(classes: string[], max = 4): string {
  if (classes.length === 0) return '—'
  if (classes.length <= max) return classes.join(', ')
  return `${classes.slice(0, max).join(', ')} +${classes.length - max}`
}

/** Stable key for pipeline topology + node params (ignores drag positions). */
export function pipelineStructureKey(nodes: AppNode[], edges: AppEdge[]): string {
  return JSON.stringify({
    nodes: nodes.map((n) => ({ id: n.id, type: n.type, data: n.data })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
    })),
  })
}

export function computeTabularModelInfo(
  dataset: LoadedDataset,
  targetColumn: string | null,
  options?: {
    pipelineNodes?: AppNode[]
    pipelineEdges?: AppEdge[]
    usePipeline?: boolean
    previewLimit?: number
    /** Cap rows when previewing pipeline stats in the UI (training uses full data). */
    pipelineMaxRows?: number
  },
): TabularModelInfo {
  const previewLimit = options?.previewLimit
  const numericFeatures = dataset.columns.filter(
    (c) => c.type === 'number' && c.name !== targetColumn,
  ).length
  const categoricalFeatures = dataset.columns.filter(
    (c) => c.type === 'string' && c.name !== targetColumn,
  ).length

  const base = {
    totalRows: dataset.rows.length,
    previewRows:
      previewLimit != null && dataset.rows.length > previewLimit ? previewLimit : null,
    numericFeatures,
    categoricalFeatures,
    targetColumn,
  }

  if (
    options?.usePipeline &&
    targetColumn &&
    options.pipelineNodes &&
    options.pipelineEdges
  ) {
    const result = executePipeline(
      dataset,
      targetColumn,
      options.pipelineNodes,
      options.pipelineEdges,
      options.pipelineMaxRows != null ? { maxRows: options.pipelineMaxRows } : undefined,
    )
    if (result.ok) {
      const d = result.data
      const targetCol = dataset.columns.find((c) => c.name === targetColumn)
      const taskType = inferTaskType(targetCol, d.classCount)
      return {
        ...base,
        featureCount: d.featureCount,
        taskType,
        classCount: taskType === 'classification' ? d.classCount : null,
        classPreview:
          taskType === 'classification' ? formatClassPreview(d.classNames) : null,
        inputShape: `${d.featureCount}×1×1`,
        outputClasses: taskType === 'classification' ? d.classCount : null,
        trainSamples: d.trainSamples,
        valSamples: d.valSamples,
        source: 'pipeline',
        pipelineError: null,
        missingTargets: 0,
      }
    }
    return {
      ...base,
      featureCount: numericFeatures,
      taskType: 'unset',
      classCount: null,
      classPreview: null,
      inputShape: numericFeatures > 0 ? `${numericFeatures}×1×1` : null,
      outputClasses: null,
      trainSamples: null,
      valSamples: null,
      source: 'raw',
      pipelineError: result.error,
      missingTargets: 0,
    }
  }

  if (!targetColumn) {
    return {
      ...base,
      featureCount: numericFeatures,
      taskType: 'unset',
      classCount: null,
      classPreview: null,
      inputShape: numericFeatures > 0 ? `${numericFeatures}×1×1` : null,
      outputClasses: null,
      trainSamples: null,
      valSamples: null,
      source: 'raw',
      pipelineError: null,
      missingTargets: 0,
    }
  }

  const targetCol = dataset.columns.find((c) => c.name === targetColumn)
  const stats = targetStats(dataset, targetColumn)
  const taskType = inferTaskType(targetCol, stats.count)
  const featureCount = numericFeatures

  return {
    ...base,
    featureCount,
    taskType,
    classCount: taskType === 'classification' ? stats.count : null,
    classPreview:
      taskType === 'classification' ? formatClassPreview(stats.unique) : null,
    inputShape: featureCount > 0 ? `${featureCount}×1×1` : null,
    outputClasses: taskType === 'classification' ? stats.count : null,
    trainSamples: null,
    valSamples: null,
    source: 'raw',
    pipelineError: null,
    missingTargets: stats.missing,
  }
}
