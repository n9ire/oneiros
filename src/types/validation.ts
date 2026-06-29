export type ValidationSeverity = 'error' | 'warning'

export interface ValidationIssue {
  nodeId: string
  severity: ValidationSeverity
  message: string
}

export interface ValidationResult {
  issues: ValidationIssue[]
  isValid: boolean
}
