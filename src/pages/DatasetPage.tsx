import { useEffect, useRef, useState } from 'react'
import { useDatasetStore } from '../store/useDatasetStore'
import type { ColumnInfo } from '../store/useDatasetStore'
import DatasetFlow, { PipelinePaletteItem } from '../editor/dataset/DatasetFlow'
import { datasetNodeDefs } from '../editor/dataset/preprocessingNodes'

const PREVIEW_ROWS = 200
const API_BASE = 'http://localhost:8000'
const MAX_PLOT_ROWS = 4000

// ── Icons ─────────────────────────────────────────────────────────────────────

function UploadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

function TableIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3h18v18H3z" />
      <path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
    </svg>
  )
}

function FlowIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="6" height="6" rx="1" />
      <rect x="16" y="7" width="6" height="6" rx="1" />
      <line x1="8" y1="10" x2="16" y2="10" />
    </svg>
  )
}

function ChartIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" /><line x1="2" y1="20" x2="22" y2="20" />
    </svg>
  )
}

function TrashIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
    </svg>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function DatasetPage() {
  const { dataset, loadFromCSV, loadFromJSON, clearDataset, targetColumn, setTargetColumn } = useDatasetStore()
  const [activeTab, setActiveTab] = useState<'table' | 'pipeline' | 'visualize'>('table')
  const csvInputRef = useRef<HTMLInputElement>(null)
  const jsonInputRef = useRef<HTMLInputElement>(null)

  async function handleCSV(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) await loadFromCSV(file)
    e.target.value = ''
  }

  async function handleJSON(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) await loadFromJSON(file)
    e.target.value = ''
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#09090b' }}>

      {/* Toolbar */}
      <div style={{
        height: 48,
        background: '#111113',
        borderBottom: '1px solid #1e1e2e',
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        gap: 10,
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: '#52525b' }}>
          Dataset Studio
        </span>

        {dataset && (
          <>
            <span style={{ fontSize: 11, color: '#3f3f46', margin: '0 2px' }}>›</span>
            <span style={{ fontSize: 12, color: '#a1a1aa', fontWeight: 500 }}>{dataset.name}</span>
            <span style={{
              fontSize: 10, color: '#52525b',
              background: '#18181b', border: '1px solid #27272a',
              borderRadius: 4, padding: '1px 7px',
            }}>
              {dataset.rows.length.toLocaleString()} rows · {dataset.columns.length} cols
            </span>
          </>
        )}

        <div style={{ flex: 1 }} />

        {dataset && (
          <>
            {/* Target column selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#52525b', whiteSpace: 'nowrap' }}>
                Target
              </span>
              <select
                value={targetColumn ?? ''}
                onChange={(e) => setTargetColumn(e.target.value || null)}
                style={{
                  background: '#18181b', border: '1px solid #27272a',
                  borderRadius: 5, color: targetColumn ? '#a78bfa' : '#52525b',
                  fontSize: 11, padding: '3px 7px', outline: 'none', cursor: 'pointer',
                  maxWidth: 140,
                }}
              >
                <option value="" style={{ background: '#18181b' }}>— none —</option>
                {dataset.columns.map((col) => (
                  <option key={col.name} value={col.name} style={{ background: '#18181b' }}>
                    {col.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Tab toggle */}
            <div style={{ display: 'flex', gap: 2, background: '#18181b', border: '1px solid #27272a', borderRadius: 6, padding: 2 }}>
              <TabBtn active={activeTab === 'table'} onClick={() => setActiveTab('table')} icon={<TableIcon />} label="Table" />
              <TabBtn active={activeTab === 'pipeline'} onClick={() => setActiveTab('pipeline')} icon={<FlowIcon />} label="Pipeline" />
              <TabBtn active={activeTab === 'visualize'} onClick={() => setActiveTab('visualize')} icon={<ChartIcon />} label="Visualize" />
            </div>

            <button
              onClick={clearDataset}
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '4px 8px', borderRadius: 5,
                border: '1px solid #27272a', background: 'transparent',
                color: '#71717a', fontSize: 11, cursor: 'pointer',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = '#f87171'; e.currentTarget.style.borderColor = '#7f1d1d' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = '#71717a'; e.currentTarget.style.borderColor = '#27272a' }}
            >
              <TrashIcon /> Clear
            </button>
          </>
        )}

        <input ref={csvInputRef} type="file" accept=".csv" onChange={handleCSV} style={{ display: 'none' }} />
        <input ref={jsonInputRef} type="file" accept=".json" onChange={handleJSON} style={{ display: 'none' }} />

        <UploadButton label="Import CSV" onClick={() => csvInputRef.current?.click()} />
        <UploadButton label="Import JSON" onClick={() => jsonInputRef.current?.click()} accent />
      </div>

      {/* Body */}
      {!dataset ? (
        <EmptyState
          onImportCSV={() => csvInputRef.current?.click()}
          onImportJSON={() => jsonInputRef.current?.click()}
        />
      ) : activeTab === 'table' ? (
        <TableView dataset={dataset} />
      ) : activeTab === 'visualize' ? (
        <VisualizeView dataset={dataset} />
      ) : (
        <PipelineView />
      )}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState({ onImportCSV, onImportJSON }: { onImportCSV: () => void; onImportJSON: () => void }) {
  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: 16,
    }}>
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#27272a" strokeWidth="1.2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
      <div style={{ textAlign: 'center' }}>
        <p style={{ fontSize: 14, color: '#52525b', margin: '0 0 6px' }}>No dataset loaded</p>
        <p style={{ fontSize: 12, color: '#3f3f46', margin: 0 }}>Import a CSV or JSON file to begin</p>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <UploadButton label="Import CSV" onClick={onImportCSV} />
        <UploadButton label="Import JSON" onClick={onImportJSON} accent />
      </div>
    </div>
  )
}

// ── Table view ────────────────────────────────────────────────────────────────

function TableView({ dataset }: { dataset: NonNullable<ReturnType<typeof useDatasetStore.getState>['dataset']> }) {
  const preview = dataset.rows.slice(0, PREVIEW_ROWS)
  const cols = dataset.columns

  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

      {/* Column stats sidebar */}
      <div style={{
        width: 220, background: '#111113',
        borderRight: '1px solid #1e1e2e',
        overflowY: 'auto', flexShrink: 0,
      }}>
        <div style={{ padding: '10px 12px 6px', fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: '#52525b' }}>
          Columns
        </div>
        {cols.map((col) => <ColumnCard key={col.name} col={col} />)}
      </div>

      {/* Data table */}
      <div style={{ flex: 1, overflow: 'auto', position: 'relative' }}>
        {dataset.rows.length > PREVIEW_ROWS && (
          <div style={{ padding: '6px 14px', background: '#1a1a2e', borderBottom: '1px solid #1e1e2e', fontSize: 11, color: '#7c3aed' }}>
            Showing first {PREVIEW_ROWS} of {dataset.rows.length.toLocaleString()} rows
          </div>
        )}
        <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#111113', position: 'sticky', top: 0, zIndex: 1 }}>
              <th style={thStyle}>#</th>
              {cols.map((col) => (
                <th key={col.name} style={thStyle}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <TypeDot type={col.type} />
                    {col.name}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}>
                <td style={indexCellStyle}>{i + 1}</td>
                {cols.map((col) => {
                  const val = row[col.name]
                  const isNull = val === null || val === undefined || val === ''
                  return (
                    <td key={col.name} style={{ ...tdStyle, color: isNull ? '#3f3f46' : col.type === 'number' ? '#a5b4fc' : '#d4d4d8' }}>
                      {isNull ? 'null' : String(val)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Pipeline view ─────────────────────────────────────────────────────────────

function PipelineView() {
  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      {/* Mini palette */}
      <div style={{
        width: 190, background: '#111113',
        borderRight: '1px solid #1e1e2e',
        padding: '10px 0',
        flexShrink: 0,
        overflowY: 'auto',
      }}>
        <div style={{ padding: '2px 12px 6px', fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: '#52525b' }}>
          Transforms
        </div>
        {datasetNodeDefs.map((def) => (
          <PipelinePaletteItem key={def.type} def={def} />
        ))}
        <div style={{ padding: '10px 12px 4px', marginTop: 8, borderTop: '1px solid #1e1e2e', fontSize: 10, color: '#3f3f46', lineHeight: 1.5 }}>
          Drag transforms onto the canvas
        </div>
      </div>

      {/* Pipeline canvas */}
      <div style={{ flex: 1, position: 'relative' }}>
        <DatasetFlow />
      </div>
    </div>
  )
}

// ── Visualize view ────────────────────────────────────────────────────────────

interface ChartDef {
  id: string
  label: string
  needsY: boolean
  allowHue: boolean
  numX: boolean
  numY: boolean
}

const CHART_GROUPS = [
  { label: 'Distribution', ids: ['hist', 'kde', 'count'] },
  { label: 'Relational', ids: ['scatter', 'line', 'regression', 'joint'] },
  { label: 'Categorical', ids: ['bar', 'box', 'violin', 'strip', 'swarm', 'point'] },
  { label: 'Multi-variate', ids: ['heatmap', 'pairplot'] },
]

function VisualizeView({ dataset }: { dataset: NonNullable<ReturnType<typeof useDatasetStore.getState>['dataset']> }) {
  const cols = dataset.columns
  const numCols = cols.filter((c) => c.type === 'number')

  const [chartType, setChartType] = useState('scatter')
  const [xCol, setXCol] = useState(numCols[0]?.name ?? cols[0]?.name ?? '')
  const [yCol, setYCol] = useState(numCols[1]?.name ?? numCols[0]?.name ?? '')
  const [hueCol, setHueCol] = useState('')
  const [palette, setPalette] = useState('mako')
  const [chartDefs, setChartDefs] = useState<ChartDef[]>([])
  const [palettes, setPalettes] = useState<string[]>([])

  const [imgSrc, setImgSrc] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const activeDef = chartDefs.find((d) => d.id === chartType) ?? null

  // Fetch chart metadata from backend once on mount
  useEffect(() => {
    const STATIC_DEFS: ChartDef[] = [
      { id: 'scatter', label: 'Scatter', needsY: true, allowHue: true, numX: true, numY: true },
      { id: 'line', label: 'Line', needsY: true, allowHue: true, numX: true, numY: true },
      { id: 'regression', label: 'Regression', needsY: true, allowHue: false, numX: true, numY: true },
      { id: 'bar', label: 'Bar', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'box', label: 'Box', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'violin', label: 'Violin', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'strip', label: 'Strip', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'swarm', label: 'Swarm', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'point', label: 'Point', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'count', label: 'Count', needsY: false, allowHue: true, numX: false, numY: false },
      { id: 'hist', label: 'Histogram', needsY: false, allowHue: true, numX: true, numY: false },
      { id: 'kde', label: 'KDE', needsY: false, allowHue: true, numX: true, numY: false },
      { id: 'joint', label: 'Joint', needsY: true, allowHue: false, numX: true, numY: true },
      { id: 'heatmap', label: 'Correlation Map', needsY: false, allowHue: false, numX: false, numY: false },
      { id: 'pairplot', label: 'Pair Plot', needsY: false, allowHue: true, numX: false, numY: false },
    ]
    setChartDefs(STATIC_DEFS)
    setPalettes(['mako', 'rocket', 'viridis', 'plasma', 'magma', 'Set2', 'tab10', 'cividis', 'flare', 'crest', 'icefire', 'husl'])

    fetch(`${API_BASE}/api/plot/meta`)
      .then((r) => r.json())
      .then((data: { chartDefs: ChartDef[]; palettes: string[] }) => {
        setChartDefs(data.chartDefs)
        setPalettes(data.palettes)
      })
      .catch(() => { /* use static fallback already set above */ })
  }, [])

  async function handlePlot() {
    setLoading(true)
    setError(null)
    try {
      const rows = dataset.rows.slice(0, MAX_PLOT_ROWS)
      const res = await fetch(`${API_BASE}/api/plot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rows,
          config: {
            chartType,
            xCol: xCol || null,
            yCol: (activeDef?.needsY ? yCol : null) || null,
            hueCol: hueCol || null,
            palette,
            maxRows: MAX_PLOT_ROWS,
          },
        }),
      })
      if (!res.ok) {
        const json = await res.json().catch(() => null)
        if (!json) {
          setError(`HTTP ${res.status}`)
        } else if (json.error) {
          setError(json.error)
        } else if (json.detail) {
          // FastAPI Pydantic validation errors come as detail[]
          const detail = Array.isArray(json.detail)
            ? json.detail.map((d: { msg?: string; loc?: string[] }) => `${d.loc?.join('.')} — ${d.msg}`).join('\n')
            : String(json.detail)
          setError(detail)
        } else {
          setError(JSON.stringify(json))
        }
        setLoading(false)
        return
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setImgSrc((prev) => { if (prev) URL.revokeObjectURL(prev); return url })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to reach backend')
    } finally {
      setLoading(false)
    }
  }

  function handleDownload() {
    if (!imgSrc) return
    const a = document.createElement('a')
    a.href = imgSrc
    a.download = `${chartType}_${xCol}_${yCol || 'plot'}.png`
    a.click()
  }

  const xOptions = activeDef?.numX ? numCols : cols
  const yOptions = activeDef?.numY ? numCols : cols

  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

      {/* ── Left panel: controls ─────────────────────────────────────────── */}
      <div style={{
        width: 240, background: '#111113',
        borderRight: '1px solid #1e1e2e',
        overflowY: 'auto', flexShrink: 0,
        display: 'flex', flexDirection: 'column',
      }}>
        {/* Chart type picker */}
        <div style={{ padding: '12px 12px 6px' }}>
          <VizLabel>Chart Type</VizLabel>
          {CHART_GROUPS.map((group) => {
            const defs = chartDefs.filter((d) => group.ids.includes(d.id))
            if (defs.length === 0) return null
            return (
              <div key={group.label} style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#3f3f46', marginBottom: 4 }}>
                  {group.label}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {defs.map((d) => (
                    <button
                      key={d.id}
                      onClick={() => setChartType(d.id)}
                      style={{
                        padding: '3px 9px', borderRadius: 5, fontSize: 11,
                        border: chartType === d.id ? '1px solid #7c3aed' : '1px solid #27272a',
                        background: chartType === d.id ? 'rgba(124,58,237,0.15)' : 'transparent',
                        color: chartType === d.id ? '#a78bfa' : '#71717a',
                        cursor: 'pointer', fontWeight: chartType === d.id ? 600 : 400,
                        transition: 'all 0.1s',
                      }}
                    >
                      {d.label}
                    </button>
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        <div style={{ borderTop: '1px solid #1e1e2e', padding: '12px 12px', display: 'flex', flexDirection: 'column', gap: 10 }}>

          {/* X Axis */}
          {activeDef?.id !== 'heatmap' && activeDef?.id !== 'pairplot' && (
            <VizSelect
              label={`X Axis${activeDef?.numX ? ' (numeric)' : ''}`}
              value={xCol}
              onChange={setXCol}
              options={xOptions.map((c) => c.name)}
            />
          )}

          {/* Y Axis */}
          {activeDef?.needsY && (
            <VizSelect
              label={`Y Axis${activeDef?.numY ? ' (numeric)' : ''}`}
              value={yCol}
              onChange={setYCol}
              options={yOptions.map((c) => c.name)}
            />
          )}

          {/* Hue */}
          {activeDef?.allowHue && (
            <VizSelect
              label="Colour by (hue)"
              value={hueCol}
              onChange={setHueCol}
              options={cols.map((c) => c.name)}
              optional
            />
          )}

          {/* Palette */}
          <div>
            <VizLabel>Palette</VizLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {palettes.slice(0, 12).map((p) => (
                <button
                  key={p}
                  onClick={() => setPalette(p)}
                  title={p}
                  style={{
                    padding: '2px 7px', borderRadius: 4, fontSize: 10,
                    border: palette === p ? '1px solid #7c3aed' : '1px solid #27272a',
                    background: palette === p ? 'rgba(124,58,237,0.15)' : '#18181b',
                    color: palette === p ? '#a78bfa' : '#52525b',
                    cursor: 'pointer', transition: 'all 0.1s',
                  }}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div style={{ flex: 1 }} />

        {/* Plot button */}
        <div style={{ padding: 12, borderTop: '1px solid #1e1e2e' }}>
          {dataset.rows.length > MAX_PLOT_ROWS && (
            <p style={{ fontSize: 10, color: '#52525b', marginBottom: 8 }}>
              Sampling {MAX_PLOT_ROWS.toLocaleString()} of {dataset.rows.length.toLocaleString()} rows
            </p>
          )}
          <button
            onClick={handlePlot}
            disabled={loading}
            style={{
              width: '100%', padding: '8px 0', borderRadius: 7,
              border: '1px solid #7c3aed',
              background: loading ? '#1a1a2e' : 'rgba(124,58,237,0.15)',
              color: loading ? '#3f3f46' : '#a78bfa',
              fontSize: 13, fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'background 0.12s',
            }}
            onMouseEnter={(e) => { if (!loading) e.currentTarget.style.background = 'rgba(124,58,237,0.28)' }}
            onMouseLeave={(e) => { if (!loading) e.currentTarget.style.background = 'rgba(124,58,237,0.15)' }}
          >
            {loading ? 'Generating…' : '⟡ Plot'}
          </button>
        </div>
      </div>

      {/* ── Right panel: chart display ───────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#09090b' }}>

        {/* Toolbar above chart */}
        {imgSrc && !loading && (
          <div style={{
            height: 36, display: 'flex', alignItems: 'center',
            padding: '0 14px', gap: 8, borderBottom: '1px solid #1e1e2e',
            background: '#111113', flexShrink: 0,
          }}>
            <span style={{ fontSize: 11, color: '#52525b' }}>
              {chartType} · {xCol}{yCol ? ` × ${yCol}` : ''}{hueCol ? ` · hue: ${hueCol}` : ''}
            </span>
            <div style={{ flex: 1 }} />
            <button
              onClick={handleDownload}
              style={{
                padding: '3px 10px', borderRadius: 5, fontSize: 11,
                border: '1px solid #27272a', background: 'transparent',
                color: '#71717a', cursor: 'pointer',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = '#a78bfa'; e.currentTarget.style.borderColor = '#7c3aed' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = '#71717a'; e.currentTarget.style.borderColor = '#27272a' }}
            >
              ↓ Save PNG
            </button>
          </div>
        )}

        <div style={{ flex: 1, overflow: 'auto', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', padding: 24 }}>
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, marginTop: 80 }}>
              <Spinner />
              <span style={{ fontSize: 12, color: '#52525b' }}>Generating plot…</span>
            </div>
          ) : error ? (
            <div style={{
              maxWidth: 420, padding: '14px 18px',
              background: 'rgba(239,68,68,0.08)', border: '1px solid #ef444430',
              borderRadius: 8, fontSize: 13, color: '#fca5a5', lineHeight: 1.5,
            }}>
              <strong style={{ display: 'block', marginBottom: 4 }}>Plot error</strong>
              {error}
            </div>
          ) : imgSrc ? (
            <img
              src={imgSrc}
              alt="Generated chart"
              style={{
                maxWidth: '100%', borderRadius: 10,
                border: '1px solid #1e1e2e',
                boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
              }}
            />
          ) : (
            <div style={{ textAlign: 'center', marginTop: 80 }}>
              <div style={{ fontSize: 40, marginBottom: 14, opacity: 0.2 }}>⟡</div>
              <p style={{ fontSize: 13, color: '#52525b', margin: '0 0 6px' }}>
                Configure a chart and click <strong style={{ color: '#71717a' }}>Plot</strong>
              </p>
              <p style={{ fontSize: 11, color: '#3f3f46', margin: 0 }}>
                {dataset.rows.length.toLocaleString()} rows · {cols.length} columns available
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Visualize sub-components ──────────────────────────────────────────────────

function VizLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#52525b', marginBottom: 5 }}>
      {children}
    </div>
  )
}

function VizSelect({ label, value, onChange, options, optional }: {
  label: string
  value: string
  onChange: (v: string) => void
  options: string[]
  optional?: boolean
}) {
  return (
    <div>
      <VizLabel>{label}</VizLabel>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%', background: '#18181b', border: '1px solid #27272a',
          borderRadius: 5, color: '#e4e4e7', fontSize: 11,
          padding: '4px 7px', outline: 'none', cursor: 'pointer',
        }}
      >
        {optional && <option value="" style={{ background: '#18181b' }}>— none —</option>}
        {options.map((o) => <option key={o} value={o} style={{ background: '#18181b' }}>{o}</option>)}
      </select>
    </div>
  )
}

function Spinner() {
  return (
    <div style={{
      width: 28, height: 28, borderRadius: '50%',
      border: '2px solid #27272a',
      borderTopColor: '#7c3aed',
      animation: 'spin 0.7s linear infinite',
    }} />
  )
}

// Inject spinner keyframe
if (typeof document !== 'undefined') {
  const id = 'oneiros-spinner-style'
  if (!document.getElementById(id)) {
    const s = document.createElement('style')
    s.id = id
    s.textContent = '@keyframes spin { to { transform: rotate(360deg); } }'
    document.head.appendChild(s)
  }
}

// ── Column card ───────────────────────────────────────────────────────────────

function ColumnCard({ col }: { col: ColumnInfo }) {
  return (
    <div style={{ padding: '8px 12px', borderBottom: '1px solid #1a1a20' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4 }}>
        <TypeDot type={col.type} />
        <span style={{ fontSize: 11, fontWeight: 500, color: '#d4d4d8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {col.name}
        </span>
      </div>
      <div style={{ fontSize: 10, color: '#52525b' }}>
        {col.type === 'number' && (
          <>
            <span>{col.min?.toFixed(2)} – {col.max?.toFixed(2)}</span>
            <span style={{ margin: '0 4px', color: '#27272a' }}>·</span>
            <span>μ {col.mean?.toFixed(2)}</span>
          </>
        )}
        {col.type === 'string' && (
          <span>{col.uniqueCount} unique</span>
        )}
        {col.nullCount > 0 && (
          <span style={{ color: '#f87171', marginLeft: 4 }}>{col.nullCount} null</span>
        )}
      </div>
      {col.type === 'number' && (
        <NullBar nullFrac={col.nullCount / (col.uniqueCount + col.nullCount)} />
      )}
    </div>
  )
}

function NullBar({ nullFrac }: { nullFrac: number }) {
  if (nullFrac === 0) return null
  return (
    <div style={{ marginTop: 4, height: 3, background: '#27272a', borderRadius: 2, overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${nullFrac * 100}%`, background: '#f87171', borderRadius: 2 }} />
    </div>
  )
}

function TypeDot({ type }: { type: string }) {
  const color = type === 'number' ? '#a5b4fc' : type === 'boolean' ? '#86efac' : '#fcd34d'
  return <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, flexShrink: 0, display: 'inline-block' }} />
}

// ── Table styles ──────────────────────────────────────────────────────────────

const thStyle: React.CSSProperties = {
  padding: '8px 12px',
  textAlign: 'left',
  fontWeight: 500,
  fontSize: 11,
  color: '#71717a',
  borderBottom: '1px solid #1e1e2e',
  whiteSpace: 'nowrap',
  letterSpacing: '0.02em',
}

const tdStyle: React.CSSProperties = {
  padding: '5px 12px',
  borderBottom: '1px solid rgba(255,255,255,0.03)',
  whiteSpace: 'nowrap',
  maxWidth: 200,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
}

const indexCellStyle: React.CSSProperties = {
  ...tdStyle,
  color: '#3f3f46',
  fontSize: 10,
  textAlign: 'right',
  paddingRight: 8,
  minWidth: 36,
}

// ── Shared components ─────────────────────────────────────────────────────────

function TabBtn({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 5,
        padding: '4px 9px', borderRadius: 4,
        border: 'none',
        background: active ? '#27272a' : 'transparent',
        color: active ? '#e4e4e7' : '#71717a',
        fontSize: 11, fontWeight: 500, cursor: 'pointer',
      }}
    >
      {icon}{label}
    </button>
  )
}

function UploadButton({ label, onClick, accent }: { label: string; onClick: () => void; accent?: boolean }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 5,
        padding: '5px 10px', borderRadius: 6,
        border: accent ? '1px solid #0ea5e9' : '1px solid #27272a',
        background: accent ? 'rgba(14,165,233,0.08)' : 'transparent',
        color: accent ? '#7dd3fc' : '#a1a1aa',
        fontSize: 12, fontWeight: 500, cursor: 'pointer',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = accent ? 'rgba(14,165,233,0.15)' : '#27272a'
        e.currentTarget.style.color = accent ? '#bae6fd' : '#e4e4e7'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = accent ? 'rgba(14,165,233,0.08)' : 'transparent'
        e.currentTarget.style.color = accent ? '#7dd3fc' : '#a1a1aa'
      }}
    >
      <UploadIcon />{label}
    </button>
  )
}
