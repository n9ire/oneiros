import { useState, useRef } from 'react'
import { useProjectStore } from '../store/useProjectStore'
import { useGraphStore } from '../store/useGraphStore'

function HomeIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  )
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function ZapIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  )
}

function SaveIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <polyline points="17 21 17 13 7 13 7 21" />
      <polyline points="7 3 7 8 15 8" />
    </svg>
  )
}

function FolderOpenIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  )
}

function PlayIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  )
}

function CodeIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  )
}

function SparkleIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17 5.8 21.3l2.4-7.4L2 9.4h7.6z" />
    </svg>
  )
}

function UndoIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 7v6h6" />
      <path d="M3 13a9 9 0 1 0 2.83-6.36L3 9" />
    </svg>
  )
}

function RedoIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 7v6h-6" />
      <path d="M21 13a9 9 0 1 1-2.83-6.36L21 9" />
    </svg>
  )
}

// ── Topbar ────────────────────────────────────────────────────────────────────

export type AppView = 'model' | 'dataset'

interface TopbarProps {
  view: AppView
  onViewChange: (v: AppView) => void
  codeOpen: boolean
  onToggleCode: () => void
  trainOpen: boolean
  onToggleTrain: () => void
  aiOpen: boolean
  onToggleAI: () => void
}

export default function Topbar({ view, onViewChange, codeOpen, onToggleCode, trainOpen, onToggleTrain, aiOpen, onToggleAI }: TopbarProps) {
  const { name, isDirty, setName, saveToFile, loadFromFile, goHome } = useProjectStore()
  const canUndo = useGraphStore((s) => s.canUndo)
  const canRedo = useGraphStore((s) => s.canRedo)
  const undo = useGraphStore((s) => s.undo)
  const redo = useGraphStore((s) => s.redo)
  const validationIssues = useGraphStore((s) => s.validationIssues)

  const errorCount = validationIssues.filter((i) => i.severity === 'error').length
  const warnCount = validationIssues.filter((i) => i.severity === 'warning').length

  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(name)
  const inputRef = useRef<HTMLInputElement>(null)

  function startEdit() {
    setDraft(name)
    setEditing(true)
    setTimeout(() => inputRef.current?.select(), 10)
  }

  function commitEdit() {
    const trimmed = draft.trim()
    if (trimmed) setName(trimmed)
    setEditing(false)
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') commitEdit()
    if (e.key === 'Escape') setEditing(false)
  }

  return (
    <header
      style={{
        height: 52,
        background: '#0d0e14',
        borderBottom: '1px solid #1e1e2e',
        display: 'flex',
        alignItems: 'center',
        padding: '0 14px',
        gap: 10,
        flexShrink: 0,
        zIndex: 10,
      }}
    >
      {/* Back to projects */}
      <button
        onClick={goHome}
        title="All Projects"
        style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '4px 7px', borderRadius: 5,
          border: 'none', background: 'transparent',
          color: '#52525b', fontSize: 11, cursor: 'pointer',
          transition: 'color 0.1s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.color = '#a1a1aa' }}
        onMouseLeave={(e) => { e.currentTarget.style.color = '#52525b' }}
      >
        <HomeIcon />
      </button>

      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#7c3aed' }}>
        <ZapIcon />
        <span style={{ fontSize: 14, fontWeight: 700, letterSpacing: '-0.02em', color: '#e4e4e7' }}>
          oneiros
        </span>
      </div>

      <Divider />

      {/* Project name */}
      {editing ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitEdit}
          onKeyDown={handleKeyDown}
          style={{
            background: '#18181b',
            border: '1px solid #7c3aed',
            borderRadius: 5,
            color: '#e4e4e7',
            fontSize: 13,
            padding: '3px 8px',
            outline: 'none',
            width: 180,
          }}
        />
      ) : (
        <button
          onClick={startEdit}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#d4d4d8',
            fontSize: 13,
            cursor: 'text',
            padding: '3px 6px',
            borderRadius: 5,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
          title="Click to rename project"
        >
          {name}
          {isDirty && (
            <span
              style={{
                width: 5,
                height: 5,
                borderRadius: '50%',
                background: '#7c3aed',
                display: 'inline-block',
                flexShrink: 0,
              }}
            />
          )}
        </button>
      )}

      <Divider />

      {/* View switcher */}
      <div style={{ display: 'flex', gap: 2, background: '#18181b', border: '1px solid #27272a', borderRadius: 6, padding: 2 }}>
        <ViewTab active={view === 'model'} onClick={() => onViewChange('model')} label="Model" />
        <ViewTab active={view === 'dataset'} onClick={() => onViewChange('dataset')} label="Dataset" />
      </div>

      <Divider />

      {/* Undo / Redo */}
      <div style={{ display: 'flex', gap: 2 }}>
        <IconButton
          onClick={undo}
          disabled={!canUndo}
          title="Undo (⌘Z)"
          icon={<UndoIcon />}
        />
        <IconButton
          onClick={redo}
          disabled={!canRedo}
          title="Redo (⌘⇧Z)"
          icon={<RedoIcon />}
        />
      </div>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Validation summary */}
      {(errorCount > 0 || warnCount > 0) && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {errorCount > 0 && (
            <ValidationPill count={errorCount} type="error" />
          )}
          {warnCount > 0 && (
            <ValidationPill count={warnCount} type="warning" />
          )}
        </div>
      )}

      <Divider />

      {/* File + view actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <TopbarButton onClick={loadFromFile} icon={<FolderOpenIcon />} label="Load" />
        <TopbarButton onClick={saveToFile} icon={<SaveIcon />} label="Save" accent />
        {view === 'model' && (
          <>
            <Divider />
            <TopbarButton
              onClick={onToggleCode}
              icon={<CodeIcon />}
              label="Code"
              active={codeOpen}
            />
            <TopbarButton
              onClick={onToggleTrain}
              icon={<PlayIcon />}
              label="Train"
              active={trainOpen}
              accent={trainOpen}
            />
          </>
        )}
        <Divider />
        <TopbarButton
          onClick={onToggleAI}
          icon={<SparkleIcon />}
          label="AI"
          active={aiOpen}
          accent={aiOpen}
        />
      </div>
    </header>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Divider() {
  return <div style={{ width: 1, height: 20, background: '#27272a', flexShrink: 0 }} />
}

function ViewTab({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '4px 10px',
        borderRadius: 4,
        border: 'none',
        background: active ? '#27272a' : 'transparent',
        color: active ? '#e4e4e7' : '#71717a',
        fontSize: 12,
        fontWeight: 500,
        cursor: 'pointer',
        transition: 'background 0.1s, color 0.1s',
      }}
      onMouseEnter={(e) => { if (!active) e.currentTarget.style.color = '#a1a1aa' }}
      onMouseLeave={(e) => { if (!active) e.currentTarget.style.color = '#71717a' }}
    >
      {label}
    </button>
  )
}

function ValidationPill({ count, type }: { count: number; type: 'error' | 'warning' }) {
  const isError = type === 'error'
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        padding: '2px 7px',
        borderRadius: 4,
        background: isError ? 'rgba(239,68,68,0.12)' : 'rgba(245,158,11,0.12)',
        border: `1px solid ${isError ? '#ef444430' : '#f59e0b30'}`,
        fontSize: 11,
        fontWeight: 600,
        color: isError ? '#f87171' : '#fbbf24',
      }}
    >
      <span>{isError ? '✕' : '△'}</span>
      {count}
    </div>
  )
}

function IconButton({
  onClick,
  disabled,
  title,
  icon,
}: {
  onClick?: () => void
  disabled?: boolean
  title?: string
  icon: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        width: 28,
        height: 28,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 6,
        border: '1px solid transparent',
        background: 'transparent',
        color: disabled ? '#3f3f46' : '#71717a',
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'background 0.1s, color 0.1s',
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.background = '#27272a'
          e.currentTarget.style.color = '#e4e4e7'
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent'
        e.currentTarget.style.color = disabled ? '#3f3f46' : '#71717a'
      }}
    >
      {icon}
    </button>
  )
}

interface TopbarButtonProps {
  onClick?: () => void
  icon: React.ReactNode
  label: string
  accent?: boolean
  active?: boolean
  disabled?: boolean
  title?: string
}

function TopbarButton({ onClick, icon, label, accent, active, disabled, title }: TopbarButtonProps) {
  const bg = accent ? '#7c3aed1a' : active ? '#27272a' : 'transparent'
  const color = disabled ? '#3f3f46' : accent ? '#a78bfa' : active ? '#e4e4e7' : '#a1a1aa'
  const border = accent ? '1px solid #7c3aed' : active ? '1px solid #3f3f46' : '1px solid #27272a'

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '5px 10px',
        borderRadius: 6,
        border,
        background: bg,
        color,
        fontSize: 12,
        fontWeight: 500,
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'background 0.12s, color 0.12s',
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.background = accent ? '#7c3aed33' : '#27272a'
          e.currentTarget.style.color = accent ? '#c4b5fd' : '#e4e4e7'
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = bg
        e.currentTarget.style.color = color
      }}
    >
      {icon}
      {label}
    </button>
  )
}
