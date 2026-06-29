import { useGraphStore } from '../store/useGraphStore'
import { getNodeDefinition } from '../editor/registry/nodeRegistry'
import type { NodeField } from '../types/node'

export default function PropertyPanel() {
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const nodes = useGraphStore((s) => s.nodes)
  const updateNodeData = useGraphStore((s) => s.updateNodeData)

  const selectedNode = nodes.find((n) => n.id === selectedNodeId)
  const def = selectedNode ? getNodeDefinition(selectedNode.type ?? '') : undefined

  return (
    <aside
      style={{
        width: 260,
        background: '#111113',
        borderLeft: '1px solid #1e1e2e',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '13px 14px 11px',
          borderBottom: '1px solid #1e1e2e',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#52525b' }}>
          Inspector
        </span>
        {selectedNode && def && (
          <span style={{ fontSize: 11, color: '#71717a', marginLeft: 'auto' }}>{def.label}</span>
        )}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {!selectedNode ? (
          <EmptyState />
        ) : !def ? (
          <div style={{ padding: 14, fontSize: 12, color: '#52525b' }}>
            No definition found for node type "{selectedNode.type}".
          </div>
        ) : (
          <div style={{ padding: '14px 14px' }}>
            <div style={{ marginBottom: 16 }}>
              <NodeIdBadge id={selectedNode.id} />
            </div>
            {def.fields.map((field) => (
              <FieldControl
                key={field.key}
                field={field}
                value={selectedNode.data[field.key]}
                onChange={(val) => updateNodeData(selectedNode.id, { [field.key]: val })}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}

function EmptyState() {
  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
        gap: 10,
        height: '100%',
        minHeight: 200,
      }}
    >
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#27272a" strokeWidth="1.5">
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <path d="M9 9h6M9 12h6M9 15h4" strokeLinecap="round" />
      </svg>
      <p style={{ fontSize: 12, color: '#3f3f46', textAlign: 'center', margin: 0, lineHeight: 1.5 }}>
        Select a node on the canvas to inspect its properties
      </p>
    </div>
  )
}

function NodeIdBadge({ id }: { id: string }) {
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        background: '#18181b',
        border: '1px solid #27272a',
        borderRadius: 5,
        padding: '3px 8px',
        fontSize: 10,
        color: '#52525b',
        fontFamily: 'ui-monospace, monospace',
      }}
    >
      {id}
    </div>
  )
}

interface FieldControlProps {
  field: NodeField
  value: unknown
  onChange: (val: unknown) => void
}

function FieldControl({ field, value, onChange }: FieldControlProps) {
  const labelStyle: React.CSSProperties = {
    display: 'block',
    fontSize: 11,
    fontWeight: 500,
    color: '#71717a',
    marginBottom: 5,
    letterSpacing: '0.02em',
  }

  const inputBaseStyle: React.CSSProperties = {
    width: '100%',
    background: '#18181b',
    border: '1px solid #27272a',
    borderRadius: 6,
    color: '#e4e4e7',
    fontSize: 12,
    padding: '6px 9px',
    outline: 'none',
    boxSizing: 'border-box',
    transition: 'border-color 0.1s',
  }

  return (
    <div style={{ marginBottom: 14 }}>
      <label style={labelStyle}>{field.label}</label>

      {field.type === 'number' && (
        <input
          type="number"
          value={typeof value === 'number' ? value : ''}
          min={field.min}
          max={field.max}
          step={field.step ?? 1}
          onChange={(e) => onChange(Number(e.target.value))}
          onFocus={(e) => (e.target.style.borderColor = '#7c3aed')}
          onBlur={(e) => (e.target.style.borderColor = '#27272a')}
          style={inputBaseStyle}
        />
      )}

      {field.type === 'text' && (
        <input
          type="text"
          value={typeof value === 'string' ? value : ''}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value)}
          onFocus={(e) => (e.target.style.borderColor = '#7c3aed')}
          onBlur={(e) => (e.target.style.borderColor = '#27272a')}
          style={inputBaseStyle}
        />
      )}

      {field.type === 'select' && (
        <select
          value={typeof value === 'string' ? value : ''}
          onChange={(e) => onChange(e.target.value)}
          onFocus={(e) => (e.target.style.borderColor = '#7c3aed')}
          onBlur={(e) => (e.target.style.borderColor = '#27272a')}
          style={{ ...inputBaseStyle, cursor: 'pointer', appearance: 'none' }}
        >
          {field.options?.map((opt) => (
            <option key={opt.value} value={opt.value} style={{ background: '#18181b' }}>
              {opt.label}
            </option>
          ))}
        </select>
      )}

      {field.type === 'boolean' && (
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 9,
            cursor: 'pointer',
          }}
        >
          <div
            onClick={() => onChange(!value)}
            style={{
              width: 32,
              height: 18,
              borderRadius: 9,
              background: value ? '#7c3aed' : '#27272a',
              position: 'relative',
              cursor: 'pointer',
              transition: 'background 0.15s',
              flexShrink: 0,
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: 3,
                left: value ? 17 : 3,
                width: 12,
                height: 12,
                borderRadius: '50%',
                background: '#e4e4e7',
                transition: 'left 0.15s',
              }}
            />
          </div>
          <span style={{ fontSize: 12, color: value ? '#d4d4d8' : '#52525b' }}>
            {value ? 'Enabled' : 'Disabled'}
          </span>
        </label>
      )}
    </div>
  )
}
