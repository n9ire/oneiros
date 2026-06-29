import type { AppNode, AppEdge } from '../../types/graph'
import type { ValidationIssue, ValidationResult } from '../../types/validation'
import { topologicalSort } from '../compiler/topoSort'

// ── Cycle detection (DFS) ─────────────────────────────────────────────────────

function detectCycles(nodes: AppNode[], edges: AppEdge[]): Set<string> {
  const adj = new Map<string, string[]>()
  for (const node of nodes) adj.set(node.id, [])
  for (const edge of edges) {
    adj.get(edge.source)?.push(edge.target)
  }

  const WHITE = 0, GRAY = 1, BLACK = 2
  const color = new Map<string, number>()
  for (const node of nodes) color.set(node.id, WHITE)

  const cycleNodes = new Set<string>()

  function dfs(id: string): boolean {
    color.set(id, GRAY)
    for (const neighbor of adj.get(id) ?? []) {
      if (color.get(neighbor) === GRAY) {
        cycleNodes.add(id)
        cycleNodes.add(neighbor)
        return true
      }
      if (color.get(neighbor) === WHITE) {
        if (dfs(neighbor)) {
          cycleNodes.add(id)
          return true
        }
      }
    }
    color.set(id, BLACK)
    return false
  }

  for (const node of nodes) {
    if (color.get(node.id) === WHITE) dfs(node.id)
  }

  return cycleNodes
}

// ── Spatial / flat tensor type tracking ───────────────────────────────────────

type TensorKind = 'spatial' | 'flat' | 'unknown'

const SPATIAL_PRODUCERS = new Set(['inputNode', 'conv2dNode', 'conv1dNode', 'maxPool2dNode', 'avgPool2dNode'])
const FLAT_PRODUCERS = new Set(['denseNode', 'flattenNode', 'adaptiveAvgPool2dNode', 'rnnNode', 'lstmNode', 'gruNode', 'transformerEncoderNode'])
const PASSTHROUGH = new Set(['dropoutNode', 'batchNormNode', 'activationNode'])
const NEEDS_SPATIAL = new Set(['conv2dNode', 'maxPool2dNode', 'avgPool2dNode'])
const NEEDS_FLAT = new Set(['denseNode', 'outputNode', 'rnnNode', 'lstmNode', 'gruNode', 'transformerEncoderNode', 'conv1dNode'])

function inferTensorKinds(
  nodes: AppNode[],
  edges: AppEdge[],
): Map<string, TensorKind> {
  const sorted = topologicalSort(nodes, edges)
  if (!sorted) return new Map()

  const parents = new Map<string, string[]>()
  for (const node of sorted) parents.set(node.id, [])
  for (const edge of edges) parents.get(edge.target)?.push(edge.source)

  const kinds = new Map<string, TensorKind>()

  for (const node of sorted) {
    const type = node.type ?? ''
    const nodeParents = parents.get(node.id) ?? []

    if (SPATIAL_PRODUCERS.has(type)) {
      kinds.set(node.id, 'spatial')
    } else if (FLAT_PRODUCERS.has(type)) {
      kinds.set(node.id, 'flat')
    } else if (PASSTHROUGH.has(type)) {
      const parentKind = nodeParents.length > 0 ? (kinds.get(nodeParents[0]) ?? 'unknown') : 'unknown'
      kinds.set(node.id, parentKind)
    } else if (type === 'outputNode') {
      kinds.set(node.id, 'flat')
    } else {
      kinds.set(node.id, 'unknown')
    }
  }

  return kinds
}

// ── Main validation entry point ───────────────────────────────────────────────

export function validateGraph(nodes: AppNode[], edges: AppEdge[]): ValidationResult {
  const issues: ValidationIssue[] = []

  const incomingCount = new Map<string, number>()
  const outgoingCount = new Map<string, number>()
  for (const node of nodes) {
    incomingCount.set(node.id, 0)
    outgoingCount.set(node.id, 0)
  }
  for (const edge of edges) {
    incomingCount.set(edge.target, (incomingCount.get(edge.target) ?? 0) + 1)
    outgoingCount.set(edge.source, (outgoingCount.get(edge.source) ?? 0) + 1)
  }

  const cycleNodes = detectCycles(nodes, edges)
  const tensorKinds = inferTensorKinds(nodes, edges)

  // Parent map for type-checking
  const parents = new Map<string, string[]>()
  for (const node of nodes) parents.set(node.id, [])
  for (const edge of edges) parents.get(edge.target)?.push(edge.source)

  for (const node of nodes) {
    const type = node.type ?? ''
    const incoming = incomingCount.get(node.id) ?? 0
    const outgoing = outgoingCount.get(node.id) ?? 0
    const nodeParents = parents.get(node.id) ?? []

    // Cycle involvement
    if (cycleNodes.has(node.id)) {
      issues.push({
        nodeId: node.id,
        severity: 'error',
        message: 'Node is part of a cycle — cycles are not allowed',
      })
    }

    // Input nodes must have no incoming edges
    if (type === 'inputNode' && incoming > 0) {
      issues.push({
        nodeId: node.id,
        severity: 'error',
        message: 'Input node must not have incoming connections',
      })
    }

    // Output nodes must have no outgoing edges
    if (type === 'outputNode' && outgoing > 0) {
      issues.push({
        nodeId: node.id,
        severity: 'error',
        message: 'Output node must not have outgoing connections',
      })
    }

    // Non-input nodes with no incoming connections
    if (type !== 'inputNode' && incoming === 0) {
      issues.push({
        nodeId: node.id,
        severity: 'warning',
        message: 'Node has no incoming connections',
      })
    }

    // Non-output nodes with no outgoing connections
    if (type !== 'outputNode' && outgoing === 0) {
      issues.push({
        nodeId: node.id,
        severity: 'warning',
        message: 'Node output is not connected',
      })
    }

    // ── Spatial / flat type-checking ──────────────────────────────────────────

    // Nodes that need a spatial input
    if (NEEDS_SPATIAL.has(type) && nodeParents.length > 0) {
      for (const pid of nodeParents) {
        const parentKind = tensorKinds.get(pid)
        if (parentKind === 'flat') {
          issues.push({
            nodeId: node.id,
            severity: 'error',
            message: `${type === 'conv2dNode' ? 'Conv2d' : 'MaxPool2d'} requires a spatial (image) input — add a Conv2d or connect from InputNode`,
          })
        }
      }
    }

    // Nodes that need a flat input
    if (NEEDS_FLAT.has(type) && nodeParents.length > 0) {
      for (const pid of nodeParents) {
        const parentKind = tensorKinds.get(pid)
        if (parentKind === 'spatial') {
          issues.push({
            nodeId: node.id,
            severity: 'error',
            message: `${type === 'denseNode' ? 'Dense' : 'Output'} node requires flat input — add a Flatten node before this layer`,
          })
        }
      }
    }

    // FlattenNode should receive spatial input (warning if flat input — redundant)
    if (type === 'flattenNode' && nodeParents.length > 0) {
      const parentKind = tensorKinds.get(nodeParents[0])
      if (parentKind === 'flat') {
        issues.push({
          nodeId: node.id,
          severity: 'warning',
          message: 'Flatten receives an already-flat tensor — this is redundant',
        })
      }
    }
  }

  return {
    issues,
    isValid: issues.every((i) => i.severity !== 'error'),
  }
}
