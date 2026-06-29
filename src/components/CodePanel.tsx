import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/useGraphStore'
import { useProjectStore } from '../store/useProjectStore'
import { compileGraph } from '../editor/compiler/compile'

interface CodePanelProps {
  onClose: () => void
}

export default function CodePanel({ onClose }: CodePanelProps) {
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const projectName = useProjectStore((s) => s.name)

  const [result, setResult] = useState<{ code: string; errors: string[]; warnings: string[] } | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    const compiled = compileGraph(nodes, edges, projectName)
    setResult(compiled)
  }, [nodes, edges, projectName])

  function handleDownload() {
    if (!result?.code) return
    const blob = new Blob([result.code], { type: 'text/x-python' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${projectName.replace(/\s+/g, '_').toLowerCase()}.py`
    a.click()
    URL.revokeObjectURL(url)
  }

  function handleCopy() {
    if (!result?.code) return
    navigator.clipboard.writeText(result.code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  const hasErrors = (result?.errors.length ?? 0) > 0

  return (
    <div
      style={{
        height: 320,
        background: '#0d0e14',
        borderTop: '1px solid #1e1e2e',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}
    >
      {/* Panel header */}
      <div
        style={{
          height: 36,
          display: 'flex',
          alignItems: 'center',
          padding: '0 14px',
          borderBottom: '1px solid #1e1e2e',
          gap: 10,
          flexShrink: 0,
        }}
      >
        <span
          style={{
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: '#52525b',
          }}
        >
          Generated PyTorch
        </span>

        {!hasErrors && result?.warnings && result.warnings.length > 0 && (
          <span
            style={{
              fontSize: 10,
              color: '#fbbf24',
              background: 'rgba(245,158,11,0.1)',
              padding: '1px 6px',
              borderRadius: 4,
            }}
          >
            {result.warnings.length} warning{result.warnings.length > 1 ? 's' : ''}
          </span>
        )}

        <div style={{ flex: 1 }} />

        {!hasErrors && (
          <>
            <PanelButton onClick={handleCopy} label={copied ? 'Copied!' : 'Copy'} />
            <PanelButton onClick={handleDownload} label="Download .py" accent />
          </>
        )}

        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#52525b',
            cursor: 'pointer',
            padding: '2px 4px',
            fontSize: 16,
            lineHeight: 1,
            borderRadius: 4,
          }}
          onMouseEnter={(e) => { e.currentTarget.style.color = '#e4e4e7' }}
          onMouseLeave={(e) => { e.currentTarget.style.color = '#52525b' }}
          title="Close"
        >
          ×
        </button>
      </div>

      {/* Code body */}
      <div style={{ flex: 1, overflow: 'auto', position: 'relative' }}>
        {hasErrors ? (
          <div style={{ padding: 16 }}>
            <p
              style={{
                fontSize: 12,
                color: '#f87171',
                marginBottom: 10,
                fontWeight: 500,
              }}
            >
              Cannot compile — fix the following errors first:
            </p>
            {result!.errors.map((err, i) => (
              <div
                key={i}
                style={{
                  fontSize: 12,
                  color: '#fca5a5',
                  padding: '5px 10px',
                  background: 'rgba(239,68,68,0.08)',
                  borderRadius: 5,
                  border: '1px solid #ef444420',
                  marginBottom: 6,
                  lineHeight: 1.5,
                }}
              >
                {err}
              </div>
            ))}
          </div>
        ) : (
          <pre
            style={{
              margin: 0,
              padding: '14px 18px',
              fontSize: 12,
              lineHeight: '1.7',
              color: '#d4d4d8',
              fontFamily: 'ui-monospace, "Cascadia Code", "Fira Code", Consolas, monospace',
              whiteSpace: 'pre',
              tabSize: 4,
            }}
          >
            {result?.code ?? ''}
          </pre>
        )}
      </div>
    </div>
  )
}

function PanelButton({
  onClick,
  label,
  accent,
}: {
  onClick: () => void
  label: string
  accent?: boolean
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '3px 9px',
        borderRadius: 5,
        border: accent ? '1px solid #7c3aed' : '1px solid #27272a',
        background: accent ? '#7c3aed1a' : 'transparent',
        color: accent ? '#a78bfa' : '#71717a',
        fontSize: 11,
        fontWeight: 500,
        cursor: 'pointer',
        transition: 'background 0.1s, color 0.1s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = accent ? '#7c3aed33' : '#27272a'
        e.currentTarget.style.color = accent ? '#c4b5fd' : '#e4e4e7'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = accent ? '#7c3aed1a' : 'transparent'
        e.currentTarget.style.color = accent ? '#a78bfa' : '#71717a'
      }}
    >
      {label}
    </button>
  )
}
