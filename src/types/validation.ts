export type ValidationSeverity = 'error' | 'warning' | 'info'

export type ValidationCategory =
  | 'structure'   // missing/extra connections, orphan nodes
  | 'shape'       // tensor shape / dimension mismatches
  | 'config'      // bad hyperparameter values
  | 'dataset'     // dataset ↔ model mismatch
  | 'training'    // runtime training errors (NaN, OOM, etc.)
  | 'other'

export interface ValidationIssue {
  nodeId?: string          // may be absent for graph-level or training issues
  severity: ValidationSeverity
  category: ValidationCategory
  message: string
  hint?: string            // actionable one-liner for the user
}

export interface ValidationResult {
  issues: ValidationIssue[]
  isValid: boolean         // true if no 'error' severity issues
}
