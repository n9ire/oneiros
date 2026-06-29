import type { AppNode, AppEdge } from '../../types/graph'
import type { ValidationIssue, ValidationResult } from '../../types/validation'
import { topologicalSort } from '../compiler/topoSort'

// ── Cycle detection ───────────────────────────────────────────────────────────

function detectCycles(nodes: AppNode[], edges: AppEdge[]): Set<string> {
  const adj = new Map<string, string[]>()
  for (const node of nodes) adj.set(node.id, [])
  for (const edge of edges) adj.get(edge.source)?.push(edge.target)

  const WHITE = 0, GRAY = 1, BLACK = 2
  const color = new Map<string, number>()
  for (const node of nodes) color.set(node.id, WHITE)
  const cycleNodes = new Set<string>()

  function dfs(id: string): boolean {
    color.set(id, GRAY)
    for (const neighbor of adj.get(id) ?? []) {
      if (color.get(neighbor) === GRAY) { cycleNodes.add(id); cycleNodes.add(neighbor); return true }
      if (color.get(neighbor) === WHITE && dfs(neighbor)) { cycleNodes.add(id); return true }
    }
    color.set(id, BLACK)
    return false
  }
  for (const node of nodes) { if (color.get(node.id) === WHITE) dfs(node.id) }
  return cycleNodes
}

// ── Tensor kind tracking ──────────────────────────────────────────────────────

type TensorKind = 'spatial' | 'flat' | 'unknown'

const SPATIAL_PRODUCERS = new Set(['inputNode', 'conv2dNode', 'conv1dNode', 'maxPool2dNode', 'avgPool2dNode'])
const FLAT_PRODUCERS    = new Set(['denseNode', 'flattenNode', 'adaptiveAvgPool2dNode', 'rnnNode', 'lstmNode', 'gruNode', 'transformerEncoderNode'])
const PASSTHROUGH       = new Set(['dropoutNode', 'batchNormNode', 'activationNode'])
const NEEDS_SPATIAL     = new Set(['conv2dNode', 'maxPool2dNode', 'avgPool2dNode'])
const NEEDS_FLAT        = new Set(['denseNode', 'outputNode', 'rnnNode', 'lstmNode', 'gruNode', 'transformerEncoderNode', 'conv1dNode'])

function inferTensorKinds(nodes: AppNode[], edges: AppEdge[]): Map<string, TensorKind> {
  const sorted = topologicalSort(nodes, edges)
  if (!sorted) return new Map()

  const parents = new Map<string, string[]>()
  for (const node of sorted) parents.set(node.id, [])
  for (const edge of edges) parents.get(edge.target)?.push(edge.source)

  const kinds = new Map<string, TensorKind>()
  for (const node of sorted) {
    const type = node.type ?? ''
    const ps = parents.get(node.id) ?? []
    if (SPATIAL_PRODUCERS.has(type)) kinds.set(node.id, 'spatial')
    else if (FLAT_PRODUCERS.has(type)) kinds.set(node.id, 'flat')
    else if (PASSTHROUGH.has(type)) kinds.set(node.id, ps.length > 0 ? (kinds.get(ps[0]) ?? 'unknown') : 'unknown')
    else if (type === 'outputNode') kinds.set(node.id, 'flat')
    else kinds.set(node.id, 'unknown')
  }
  return kinds
}

// ── Shape inference (mirrors backend compiler) ────────────────────────────────

interface FlatShape  { kind: 'flat';    features: number }
interface SpatShape  { kind: 'spatial'; channels: number; height: number; width: number }
type Shape = FlatShape | SpatShape

function flatF(s: Shape) { return s.kind === 'flat' ? s.features : s.channels * s.height * s.width }
function convDim(dim: number, k: number, s: number, p: number) { return Math.max(1, Math.floor((dim + 2 * p - k) / s + 1)) }

function inferShapes(nodes: AppNode[], edges: AppEdge[]): Map<string, Shape> {
  const sorted = topologicalSort(nodes, edges)
  if (!sorted) return new Map()

  const parents = new Map<string, string[]>()
  for (const node of sorted) parents.set(node.id, [])
  for (const edge of edges) parents.get(edge.target)?.push(edge.source)

  const shapes = new Map<string, Shape>()

  for (const node of sorted) {
    const type = node.type ?? ''
    const d = node.data as Record<string, unknown>
    const nid = node.id
    const ps = parents.get(nid) ?? []
    const p0 = ps.length > 0 ? shapes.get(ps[0]) : undefined

    if (type === 'inputNode') {
      shapes.set(nid, { kind: 'spatial', channels: Number(d.channels ?? 1), height: Number(d.height ?? 28), width: Number(d.width ?? 28) })
    } else if (type === 'conv2dNode') {
      const inC = p0?.kind === 'spatial' ? p0.channels : 1
      const ih  = p0?.kind === 'spatial' ? p0.height : 1
      const iw  = p0?.kind === 'spatial' ? p0.width : 1
      const k = Number(d.kernelSize ?? 3), s = Number(d.stride ?? 1), pad = Number(d.padding ?? 0)
      shapes.set(nid, { kind: 'spatial', channels: Number(d.outChannels ?? 32), height: convDim(ih, k, s, pad), width: convDim(iw, k, s, pad) })
      void inC
    } else if (type === 'maxPool2dNode' || type === 'avgPool2dNode') {
      const poolC = p0?.kind === 'spatial' ? p0.channels : 1
      const ih    = p0?.kind === 'spatial' ? p0.height : 1
      const iw    = p0?.kind === 'spatial' ? p0.width : 1
      const k = Number(d.kernelSize ?? 2), s = Number(d.stride ?? k), pad = Number(d.padding ?? 0)
      shapes.set(nid, { kind: 'spatial', channels: poolC, height: convDim(ih, k, s, pad), width: convDim(iw, k, s, pad) })
    } else if (type === 'flattenNode') {
      shapes.set(nid, { kind: 'flat', features: p0 ? flatF(p0) : 0 })
    } else if (type === 'denseNode') {
      const inF = ps.reduce((acc, pid) => acc + (shapes.has(pid) ? flatF(shapes.get(pid)!) : 0), 0)
      shapes.set(nid, { kind: 'flat', features: Number(d.units ?? 128) })
      shapes.set(`${nid}__in`, { kind: 'flat', features: inF })
    } else if (type === 'outputNode') {
      const inF = ps.reduce((acc, pid) => acc + (shapes.has(pid) ? flatF(shapes.get(pid)!) : 0), 0)
      shapes.set(nid, { kind: 'flat', features: Number(d.classes ?? 10) })
      shapes.set(`${nid}__in`, { kind: 'flat', features: inF })
    } else if (PASSTHROUGH.has(type) && p0) {
      shapes.set(nid, p0)
    } else if (type === 'rnnNode' || type === 'gruNode' || type === 'lstmNode') {
      const h = Number(d.hiddenSize ?? 128), bidir = Boolean(d.bidirectional)
      shapes.set(nid, { kind: 'flat', features: h * (bidir ? 2 : 1) })
    } else if (type === 'transformerEncoderNode') {
      shapes.set(nid, { kind: 'flat', features: Number(d.dModel ?? 256) })
    } else if (type === 'adaptiveAvgPool2dNode') {
      const ic = p0?.kind === 'spatial' ? p0.channels : 1
      const sz = Number(d.outputSize ?? 1)
      shapes.set(nid, sz === 1 ? { kind: 'flat', features: ic } : { kind: 'spatial', channels: ic, height: sz, width: sz })
    }
  }
  return shapes
}

// ── Human-readable node label ─────────────────────────────────────────────────

function nodeLabel(type: string, data: Record<string, unknown>): string {
  const labels: Record<string, string> = {
    inputNode: 'Input', outputNode: 'Output', denseNode: 'Dense', conv2dNode: 'Conv2d',
    conv1dNode: 'Conv1d', maxPool2dNode: 'MaxPool2d', avgPool2dNode: 'AvgPool2d',
    flattenNode: 'Flatten', dropoutNode: 'Dropout', batchNormNode: 'BatchNorm',
    activationNode: 'Activation', rnnNode: 'RNN', lstmNode: 'LSTM', gruNode: 'GRU',
    transformerEncoderNode: 'TransformerEncoder', adaptiveAvgPool2dNode: 'AdaptiveAvgPool2d',
  }
  return (typeof data.label === 'string' && data.label) ? data.label : (labels[type] ?? type)
}

// ── Main entry point ──────────────────────────────────────────────────────────

export interface GraphValidationOptions {
  /** If provided, used to validate Input node dimensions against dataset */
  customFeatureCount?: number | null
  /** If provided, used to validate Output node class count against dataset */
  customClassCount?: number | null
}

export function validateGraph(
  nodes: AppNode[],
  edges: AppEdge[],
  opts: GraphValidationOptions = {},
): ValidationResult {
  const issues: ValidationIssue[] = []

  if (nodes.length === 0) {
    issues.push({ severity: 'error', category: 'structure', message: 'Graph is empty — add at least an Input and Output node.' })
    return { issues, isValid: false }
  }

  const inCount  = new Map<string, number>()
  const outCount = new Map<string, number>()
  for (const n of nodes) { inCount.set(n.id, 0); outCount.set(n.id, 0) }
  for (const e of edges) {
    inCount.set(e.target,  (inCount.get(e.target)  ?? 0) + 1)
    outCount.set(e.source, (outCount.get(e.source) ?? 0) + 1)
  }

  const cycleNodes  = detectCycles(nodes, edges)
  const tensorKinds = inferTensorKinds(nodes, edges)
  const shapes      = inferShapes(nodes, edges)

  const parents = new Map<string, string[]>()
  for (const n of nodes) parents.set(n.id, [])
  for (const e of edges) parents.get(e.target)?.push(e.source)

  const inputNodes  = nodes.filter(n => n.type === 'inputNode')
  const outputNodes = nodes.filter(n => n.type === 'outputNode')

  // ── Graph-level checks ────────────────────────────────────────────────────

  if (inputNodes.length === 0) {
    issues.push({ severity: 'error', category: 'structure',
      message: 'No Input node found.',
      hint: 'Drag an Input node from the sidebar and connect it to the first layer.' })
  }
  if (outputNodes.length === 0) {
    issues.push({ severity: 'error', category: 'structure',
      message: 'No Output node found.',
      hint: 'Drag an Output node from the sidebar and connect it to the last Dense layer.' })
  }
  if (inputNodes.length > 1) {
    issues.push({ severity: 'warning', category: 'structure',
      message: `${inputNodes.length} Input nodes found — only the first is used.` })
  }

  // ── Per-node checks ───────────────────────────────────────────────────────

  for (const node of nodes) {
    const type  = node.type ?? ''
    const data  = (node.data ?? {}) as Record<string, unknown>
    const nid   = node.id
    const label = nodeLabel(type, data)
    const incoming = inCount.get(nid)  ?? 0
    const outgoing = outCount.get(nid) ?? 0
    const ps       = parents.get(nid)  ?? []

    // Cycles
    if (cycleNodes.has(nid)) {
      issues.push({ nodeId: nid, severity: 'error', category: 'structure',
        message: `"${label}" is part of a cycle.`,
        hint: 'Remove the connection that creates the loop.' })
    }

    // Input has incoming edges
    if (type === 'inputNode' && incoming > 0) {
      issues.push({ nodeId: nid, severity: 'error', category: 'structure',
        message: `Input node "${label}" has an incoming connection — it must be the first node.`,
        hint: 'Delete the incoming edge on this Input node.' })
    }

    // Output has outgoing edges
    if (type === 'outputNode' && outgoing > 0) {
      issues.push({ nodeId: nid, severity: 'error', category: 'structure',
        message: `Output node "${label}" has an outgoing connection — it must be the last node.`,
        hint: 'Delete the outgoing edge on this Output node.' })
    }

    // Disconnected non-input node
    if (type !== 'inputNode' && incoming === 0) {
      issues.push({ nodeId: nid, severity: 'warning', category: 'structure',
        message: `"${label}" has no incoming connection.`,
        hint: 'Connect a preceding layer to this node, or remove it if unused.' })
    }

    // Disconnected non-output node
    if (type !== 'outputNode' && outgoing === 0) {
      issues.push({ nodeId: nid, severity: 'warning', category: 'structure',
        message: `"${label}" output is not connected to anything.`,
        hint: 'Connect this node to the next layer, or remove it if unused.' })
    }

    // ── Spatial / flat type checks ─────────────────────────────────────────

    if (NEEDS_SPATIAL.has(type) && ps.length > 0) {
      for (const pid of ps) {
        if (tensorKinds.get(pid) === 'flat') {
          const name = type === 'conv2dNode' ? 'Conv2d' : type === 'maxPool2dNode' ? 'MaxPool2d' : 'AvgPool2d'
          issues.push({ nodeId: nid, severity: 'error', category: 'shape',
            message: `"${label}" (${name}) requires a spatial (image) tensor but receives a flat one.`,
            hint: 'Connect from an Input node or a Conv2d layer, not from a Dense or Flatten node.' })
        }
      }
    }

    if (NEEDS_FLAT.has(type) && ps.length > 0) {
      for (const pid of ps) {
        if (tensorKinds.get(pid) === 'spatial') {
          const name = type === 'denseNode' ? 'Dense' : type === 'outputNode' ? 'Output' : type
          issues.push({ nodeId: nid, severity: 'error', category: 'shape',
            message: `"${label}" (${name}) requires a flat tensor but receives a spatial one.`,
            hint: 'Add a Flatten node between the previous layer and this one.' })
        }
      }
    }

    if (type === 'flattenNode' && ps.length > 0 && tensorKinds.get(ps[0]) === 'flat') {
      issues.push({ nodeId: nid, severity: 'warning', category: 'shape',
        message: `"${label}" (Flatten) is receiving an already-flat tensor — this step is redundant.` })
    }

    // ── Config checks ──────────────────────────────────────────────────────

    if (type === 'conv2dNode') {
      const outC  = Number(data.outChannels ?? 32)
      const grp   = Number(data.groups ?? 1)
      const shape = shapes.get(nid)
      const inC   = ps.length > 0 ? (() => { const s = shapes.get(ps[0]); return s?.kind === 'spatial' ? s.channels : 1 })() : 1
      if (grp > 1 && inC % grp !== 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `"${label}" groups=${grp} doesn't divide in_channels=${inC}.`,
          hint: `Set groups to a value that divides both ${inC} (in_channels) and ${outC} (out_channels).` })
      }
      if (grp > 1 && outC % grp !== 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `"${label}" groups=${grp} doesn't divide out_channels=${outC}.`,
          hint: `Set groups to a value that divides both ${inC} (in_channels) and ${outC} (out_channels).` })
      }
      if (shape?.kind === 'spatial' && (shape.height <= 0 || shape.width <= 0)) {
        issues.push({ nodeId: nid, severity: 'error', category: 'shape',
          message: `"${label}" produces zero-size output (kernel too large for input).`,
          hint: 'Reduce kernel_size or stride, or add padding.' })
      }
    }

    if (type === 'denseNode' || type === 'outputNode') {
      const inShape  = shapes.get(`${nid}__in`)
      const outShape = shapes.get(nid)
      if (inShape && flatF(inShape) === 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'shape',
          message: `"${label}" has zero input features — nothing is connected.`,
          hint: 'Connect a layer that produces output (Dense, Flatten, etc.) to this node.' })
      }
      if (type === 'outputNode' && outShape && flatF(outShape) === 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `"${label}" has 0 output classes.`,
          hint: 'Set "classes" to the number of classes in your dataset (e.g. 10 for MNIST).' })
      }
    }

    if (type === 'dropoutNode') {
      const p = Number(data.p ?? 0.5)
      if (p < 0 || p >= 1) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `"${label}" dropout probability ${p} is out of range.`,
          hint: 'Set p to a value in [0, 1).' })
      }
    }

    if ((type === 'rnnNode' || type === 'lstmNode' || type === 'gruNode') && Number(data.numLayers ?? 1) > 1 && Number(data.dropout ?? 0) === 0) {
      issues.push({ nodeId: nid, severity: 'info', category: 'config',
        message: `"${label}" has ${data.numLayers} layers with no dropout between them.`,
        hint: 'Consider setting dropout > 0 to regularise a deep RNN.' })
    }

    // ── Input node ↔ dataset checks ────────────────────────────────────────

    if (type === 'inputNode') {
      const ch = Number(data.channels ?? 1)
      const h  = Number(data.height   ?? 28)
      const w  = Number(data.width    ?? 28)
      const totalFeatures = ch * h * w

      if (opts.customFeatureCount != null) {
        if (totalFeatures !== opts.customFeatureCount) {
          issues.push({ nodeId: nid, severity: 'error', category: 'dataset',
            message: `Input node has ${totalFeatures} features (${ch}×${h}×${w}) but your CSV dataset has ${opts.customFeatureCount} features.`,
            hint: `Set channels=${opts.customFeatureCount}, height=1, width=1 on the Input node to match your data.` })
        }
      }
    }

    // ── Output node ↔ dataset checks ──────────────────────────────────────

    if (type === 'outputNode' && opts.customClassCount != null) {
      const classes = Number(data.classes ?? 10)
      if (classes !== opts.customClassCount) {
        issues.push({ nodeId: nid, severity: 'warning', category: 'dataset',
          message: `Output node has ${classes} classes but your dataset has ${opts.customClassCount} unique target values.`,
          hint: `Set "classes" on the Output node to ${opts.customClassCount}.` })
      }
    }
  }

  return {
    issues,
    isValid: issues.every(i => i.severity !== 'error'),
  }
}
