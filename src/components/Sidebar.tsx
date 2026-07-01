import { useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { getAllNodes } from '../editor/registry/nodeRegistry'
import '../editor/nodes/index'
import type { NodeCategory } from '../types/node'
import { COLLAPSED_PANEL_WIDTH, CollapseBtn, CollapsedBar } from './panelChrome'

const categoryLabels: Record<NodeCategory, string> = {
  input: 'Input / Output',
  output: 'Input / Output',
  layers: 'Layers',
  activation: 'Activation & Norm',
  recurrent: 'Recurrent',
  attention: 'Attention',
}

const categoryOrder: NodeCategory[] = ['input', 'layers', 'recurrent', 'attention', 'activation']

const categoryAccent: Record<NodeCategory, string> = {
  input: '#3b82f6',
  output: '#10b981',
  layers: '#8b5cf6',
  activation: '#f59e0b',
  recurrent: '#ec4899',
  attention: '#06b6d4',
}

function SearchIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  )
}

const COLLAPSED_WIDTH = COLLAPSED_PANEL_WIDTH
const MIN_WIDTH = 160
const MAX_WIDTH = 420

export default function Sidebar() {
  const [query, setQuery] = useState('')
  const [width, setWidth] = useState(220)
  const [collapsed, setCollapsed] = useState(false)
  const drag = useRef({ active: false, startX: 0, startW: 0 })
  const allNodes = getAllNodes()
  const panelWidth = collapsed ? COLLAPSED_WIDTH : width

  const filtered = query.trim()
    ? allNodes.filter(
        (n) =>
          n.label.toLowerCase().includes(query.toLowerCase()) ||
          n.description.toLowerCase().includes(query.toLowerCase())
      )
    : allNodes

  const grouped: Partial<Record<string, typeof allNodes>> = {}
  for (const node of filtered) {
    const section = node.category === 'output' ? 'input' : node.category
    if (!grouped[section]) grouped[section] = []
    grouped[section]!.push(node)
  }

  function onDragStart(e: DragEvent<HTMLDivElement>, type: string) {
    e.dataTransfer.setData('application/oneiros-node', type)
    e.dataTransfer.effectAllowed = 'copy'
  }

  function onResizeMouseDown(e: React.MouseEvent) {
    if (collapsed) return
    e.preventDefault()
    drag.current = { active: true, startX: e.clientX, startW: width }
    const onMove = (ev: MouseEvent) => {
      if (!drag.current.active) return
      const next = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, drag.current.startW + ev.clientX - drag.current.startX))
      setWidth(next)
    }
    const onUp = () => {
      drag.current.active = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  return (
    <aside
      style={{
        width: panelWidth,
        minWidth: panelWidth,
        background: '#111113',
        borderRight: '1px solid #1e1e2e',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        overflow: 'hidden',
        position: 'relative',
        transition: 'width 0.18s ease, min-width 0.18s ease',
      }}
    >
      {collapsed ? (
        <CollapsedBar
          side="left"
          label="Nodes"
          onToggle={() => setCollapsed(false)}
        />
      ) : (
        <>
      {/* Header */}
      <div style={{ padding: '12px 12px 8px', borderBottom: '1px solid #1e1e2e', display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 7, flex: 1, minWidth: 0,
          background: '#18181b', border: '1px solid #27272a',
          borderRadius: 7, padding: '6px 10px',
        }}>
          <span style={{ color: '#52525b' }}><SearchIcon /></span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search nodes…"
            style={{ background: 'transparent', border: 'none', outline: 'none', color: '#d4d4d8', fontSize: 12, flex: 1, minWidth: 0 }}
          />
        </div>
        <CollapseBtn side="left" onClick={() => setCollapsed(true)} title="Collapse palette" />
      </div>

      {/* Node palette */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
        {categoryOrder.map((section) => {
          const items = grouped[section]
          if (!items || items.length === 0) return null
          return (
            <div key={section} style={{ marginBottom: 4 }}>
              <div style={{ padding: '6px 14px 4px', fontSize: 10, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#52525b' }}>
                {categoryLabels[section]}
              </div>
              {items.map((node) => (
                <PaletteItem
                  key={node.type}
                  label={node.label}
                  description={node.description}
                  color={categoryAccent[node.category]}
                  onDragStart={(e) => onDragStart(e, node.type)}
                />
              ))}
            </div>
          )
        })}
        {filtered.length === 0 && (
          <div style={{ padding: '24px 14px', textAlign: 'center', color: '#3f3f46', fontSize: 12 }}>
            No nodes match "{query}"
          </div>
        )}
      </div>

      {/* Footer hint */}
      <div style={{ padding: '8px 14px', borderTop: '1px solid #1e1e2e', fontSize: 11, color: '#3f3f46', lineHeight: 1.5 }}>
        Drag nodes onto the canvas
      </div>

      {/* Resize handle */}
      <div
        onMouseDown={onResizeMouseDown}
        title="Drag to resize"
        style={{
          position: 'absolute', top: 0, right: 0, width: 4, height: '100%',
          cursor: 'ew-resize', zIndex: 10,
          background: 'transparent',
          transition: 'background 0.15s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = '#7c3aed66' }}
        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
      />
        </>
      )}
    </aside>
  )
}

interface PaletteItemProps {
  label: string
  description: string
  color: string
  onDragStart: (e: DragEvent<HTMLDivElement>) => void
}

function PaletteItem({ label, description, color, onDragStart }: PaletteItemProps) {
  return (
    <div
      draggable
      onDragStart={onDragStart}
      style={{
        margin: '2px 8px', padding: '8px 10px', borderRadius: 7,
        border: '1px solid transparent', cursor: 'grab',
        display: 'flex', alignItems: 'center', gap: 9,
        transition: 'background 0.1s, border-color 0.1s', userSelect: 'none',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = '#18181b'; e.currentTarget.style.borderColor = '#27272a' }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'transparent' }}
    >
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 500, color: '#d4d4d8', lineHeight: 1.3 }}>{label}</div>
        <div style={{ fontSize: 10, color: '#52525b', lineHeight: 1.3, marginTop: 1 }}>{description}</div>
      </div>
    </div>
  )
}
