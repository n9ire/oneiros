import { Handle, Position } from '@xyflow/react'
import type { NodeProps, Node } from '@xyflow/react'

// ── Category colour palette ───────────────────────────────────────────────────

const C = {
  source: { border: '#0ea5e9', header: 'rgba(14,165,233,0.1)', text: '#7dd3fc', handle: '#0ea5e9' },
  transform: { border: '#6366f1', header: 'rgba(99,102,241,0.1)', text: '#a5b4fc', handle: '#6366f1' },
  output: { border: '#10b981', header: 'rgba(16,185,129,0.1)', text: '#6ee7b7', handle: '#10b981' },
}

// ── Shared components ─────────────────────────────────────────────────────────

function DSNode({
  label,
  color,
  selected = false,
  hasTarget = true,
  hasSource = true,
  children,
}: {
  label: string
  color: typeof C.source
  selected?: boolean
  hasTarget?: boolean
  hasSource?: boolean
  children?: React.ReactNode
}) {
  return (
    <div
      style={{
        width: 196,
        background: '#18181b',
        borderRadius: 9,
        border: `1px solid ${selected ? color.border : '#27272a'}`,
        boxShadow: selected
          ? `0 0 0 1px ${color.border}30, 0 6px 20px rgba(0,0,0,0.4)`
          : '0 3px 12px rgba(0,0,0,0.35)',
        overflow: 'hidden',
        transition: 'border-color 0.15s, box-shadow 0.15s',
      }}
    >
      {hasTarget && (
        <Handle type="target" position={Position.Left}
          style={{ width: 9, height: 9, background: '#18181b', border: `2px solid ${color.handle}`, left: -5, borderRadius: '50%' }} />
      )}
      <div style={{ background: color.header, borderBottom: '1px solid #27272a', padding: '7px 11px', display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: color.text }}>{label}</span>
        <span style={{ marginLeft: 'auto', width: 5, height: 5, borderRadius: '50%', background: color.border }} />
      </div>
      {children && <div style={{ padding: '8px 11px' }}>{children}</div>}
      {hasSource && (
        <Handle type="source" position={Position.Right}
          style={{ width: 9, height: 9, background: color.handle, border: `2px solid ${color.handle}80`, right: -5, borderRadius: '50%' }} />
      )}
    </div>
  )
}

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
      <span style={{ color: '#71717a' }}>{label}</span>
      <span style={{ color: '#e4e4e7', fontWeight: 500 }}>{value}</span>
    </div>
  )
}

// ── Source node ───────────────────────────────────────────────────────────────

type SourceData = Node<{ label: string; datasetName: string }, 'datasetSource'>

export function DatasetSourceNode({ data, selected }: NodeProps<SourceData>) {
  return (
    <DSNode label="Dataset Source" color={C.source} selected={selected} hasTarget={false}>
      <Row label="File" value={data.datasetName || '—'} />
    </DSNode>
  )
}

// ── Normalize node ────────────────────────────────────────────────────────────

type NormalizeData = Node<{ label: string; method: string; columns: string }, 'normalizeNode'>

export function NormalizeNode({ data, selected }: NodeProps<NormalizeData>) {
  return (
    <DSNode label="Normalize" color={C.transform} selected={selected}>
      <Row label="Method" value={data.method || 'min-max'} />
      <Row label="Columns" value={data.columns || 'all numeric'} />
    </DSNode>
  )
}

// ── One-hot encode node ───────────────────────────────────────────────────────

type OneHotData = Node<{ label: string; columns: string }, 'oneHotEncodeNode'>

export function OneHotEncodeNode({ data, selected }: NodeProps<OneHotData>) {
  return (
    <DSNode label="One-Hot Encode" color={C.transform} selected={selected}>
      <Row label="Columns" value={data.columns || 'all categorical'} />
    </DSNode>
  )
}

// ── Shuffle node ──────────────────────────────────────────────────────────────

type ShuffleData = Node<{ label: string; seed: number }, 'shuffleNode'>

export function ShuffleNode({ data, selected }: NodeProps<ShuffleData>) {
  return (
    <DSNode label="Shuffle" color={C.transform} selected={selected}>
      <Row label="Seed" value={data.seed ?? 42} />
    </DSNode>
  )
}

// ── Split node ────────────────────────────────────────────────────────────────

type SplitData = Node<{ label: string; trainRatio: number; valRatio: number; testRatio: number }, 'splitNode'>

export function SplitNode({ data, selected }: NodeProps<SplitData>) {
  const train = data.trainRatio ?? 0.7
  const val = data.valRatio ?? 0.15
  const test = data.testRatio ?? 0.15
  return (
    <DSNode label="Split" color={C.output} selected={selected} hasSource={false}>
      <Row label="Train" value={`${Math.round(train * 100)}%`} />
      <Row label="Val" value={`${Math.round(val * 100)}%`} />
      <Row label="Test" value={`${Math.round(test * 100)}%`} />
    </DSNode>
  )
}

// ── Filter node ───────────────────────────────────────────────────────────────

type FilterData = Node<{ label: string; column: string; operator: string; value: string }, 'filterNode'>

export function FilterNode({ data, selected }: NodeProps<FilterData>) {
  return (
    <DSNode label="Filter" color={C.transform} selected={selected}>
      <Row label="Column" value={data.column || '—'} />
      <Row label="Rule" value={`${data.operator || '>'} ${data.value || '0'}`} />
    </DSNode>
  )
}

// ── Node type registry for XYFlow ─────────────────────────────────────────────

export const datasetNodeTypes = {
  datasetSource: DatasetSourceNode,
  normalizeNode: NormalizeNode,
  oneHotEncodeNode: OneHotEncodeNode,
  shuffleNode: ShuffleNode,
  splitNode: SplitNode,
  filterNode: FilterNode,
}

// ── Palette definition (used by the pipeline toolbar) ────────────────────────

export interface DatasetNodeDef {
  type: string
  label: string
  description: string
  color: string
  defaultData: Record<string, unknown>
}

export const datasetNodeDefs: DatasetNodeDef[] = [
  {
    type: 'normalizeNode',
    label: 'Normalize',
    description: 'Scale numeric columns',
    color: '#6366f1',
    defaultData: { label: 'Normalize', method: 'min-max', columns: '' },
  },
  {
    type: 'oneHotEncodeNode',
    label: 'One-Hot Encode',
    description: 'Encode categorical columns',
    color: '#6366f1',
    defaultData: { label: 'One-Hot Encode', columns: '' },
  },
  {
    type: 'shuffleNode',
    label: 'Shuffle',
    description: 'Randomise row order',
    color: '#6366f1',
    defaultData: { label: 'Shuffle', seed: 42 },
  },
  {
    type: 'splitNode',
    label: 'Split',
    description: 'Train / val / test split',
    color: '#10b981',
    defaultData: { label: 'Split', trainRatio: 0.7, valRatio: 0.15, testRatio: 0.15 },
  },
  {
    type: 'filterNode',
    label: 'Filter',
    description: 'Filter rows by condition',
    color: '#6366f1',
    defaultData: { label: 'Filter', column: '', operator: '>', value: '0' },
  },
]
