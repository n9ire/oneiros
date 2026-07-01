import type { LoadedDataset } from '../../store/useDatasetStore'
import type { TrainingConfig } from '../../types/training'
import type { ValidationIssue, ValidationResult } from '../../types/validation'
import type { AppNode, AppEdge } from '../../types/graph'
import { executePipeline } from '../dataset/pipelineExecutor'
import { inferTaskType } from '../dataset/datasetModelInfo'

function targetStats(dataset: LoadedDataset, targetColumn: string) {
  const values = dataset.rows
    .map((r) => r[targetColumn])
    .filter((v) => v !== null && v !== undefined && v !== '')
  const unique = [...new Set(values.map(String))].sort()
  return { values, unique, count: unique.length, missing: dataset.rows.length - values.length }
}

export function validateTabularTraining(
  dataset: LoadedDataset | null,
  targetColumn: string | null,
  pipelineNodes: AppNode[],
  pipelineEdges: AppEdge[],
  config: Pick<TrainingConfig, 'xgbTask' | 'xgbNEstimators' | 'xgbEarlyStoppingRounds'>,
): ValidationResult {
  const issues: ValidationIssue[] = []

  if (!dataset) {
    issues.push({
      severity: 'error',
      category: 'dataset',
      message: 'No dataset loaded.',
      hint: 'Import a CSV or JSON file in the Dataset tab.',
    })
    return { issues, isValid: false }
  }

  if (!targetColumn) {
    issues.push({
      severity: 'error',
      category: 'dataset',
      message: 'No target column selected.',
      hint: 'Choose a target column in the Dataset tab toolbar.',
    })
    return { issues, isValid: false }
  }

  const targetCol = dataset.columns.find((c) => c.name === targetColumn)
  if (!targetCol) {
    issues.push({
      severity: 'error',
      category: 'dataset',
      message: `Target column "${targetColumn}" not found in dataset.`,
      hint: 'Re-select the target column — the dataset may have changed.',
    })
    return { issues, isValid: false }
  }

  const stats = targetStats(dataset, targetColumn)
  if (stats.values.length < 4) {
    issues.push({
      severity: 'error',
      category: 'dataset',
      message: `Too few labeled rows (${stats.values.length}). Need at least 4.`,
      hint: 'Add more rows or relax pipeline filters.',
    })
  }
  if (stats.missing > 0) {
    issues.push({
      severity: 'warning',
      category: 'dataset',
      message: `${stats.missing} row(s) have missing target values.`,
      hint: 'Rows with empty targets are excluded during training.',
    })
  }

  const inferredTask = inferTaskType(targetCol, stats.count)
  if (config.xgbTask !== inferredTask) {
    issues.push({
      severity: 'error',
      category: 'dataset',
      message: `Task is set to ${config.xgbTask} but target "${targetColumn}" looks like ${inferredTask}.`,
      hint:
        inferredTask === 'regression'
          ? 'Switch Task to Regression for numeric continuous targets.'
          : 'Switch Task to Classification for categorical or low-cardinality targets.',
    })
  }

  if (config.xgbTask === 'classification') {
    if (stats.count < 2) {
      issues.push({
        severity: 'error',
        category: 'dataset',
        message: `Classification needs at least 2 classes; found ${stats.count}.`,
        hint: 'Pick a target column with multiple distinct values.',
      })
    }
  } else if (config.xgbTask === 'regression') {
    const numeric = stats.values.filter((v) => typeof v === 'number' && isFinite(v as number))
    if (numeric.length < stats.values.length * 0.9) {
      issues.push({
        severity: 'error',
        category: 'dataset',
        message: 'Regression target must be mostly numeric.',
        hint: 'Use a numeric column as target or switch to Classification.',
      })
    }
  }

  if (
    config.xgbEarlyStoppingRounds > 0 &&
    config.xgbEarlyStoppingRounds >= config.xgbNEstimators
  ) {
    issues.push({
      severity: 'error',
      category: 'config',
      message: 'early_stopping_rounds must be less than n_estimators.',
      hint: 'Lower early stopping rounds or increase n_estimators.',
    })
  }

  const pipelineResult = executePipeline(
    dataset,
    targetColumn,
    pipelineNodes,
    pipelineEdges,
    { task: config.xgbTask },
  )
  if (!pipelineResult.ok) {
    issues.push({
      severity: 'error',
      category: 'dataset',
      message: pipelineResult.error,
      hint: 'Fix the preprocessing pipeline in Dataset → Pipeline.',
    })
  } else if (pipelineResult.data.featureCount === 0) {
    issues.push({
      severity: 'error',
      category: 'dataset',
      message: 'Pipeline produced zero features.',
      hint: 'Add numeric features or encode categorical columns in the pipeline.',
    })
  }

  return {
    issues,
    isValid: !issues.some((i) => i.severity === 'error'),
  }
}
