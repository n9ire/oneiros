import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { useTrainingStore } from '../store/useTrainingStore'
import { useDatasetStore } from '../store/useDatasetStore'
import type { TrainingConfig, EpochMetrics, SchedulerType } from '../types/training'

interface TrainingPanelProps {
  onClose: () => void
}

export default function TrainingPanel({ onClose }: TrainingPanelProps) {
  const status = useTrainingStore((s) => s.status)
  const statusMessage = useTrainingStore((s) => s.statusMessage)
  const config = useTrainingStore((s) => s.config)
  const epochMetrics = useTrainingStore((s) => s.epochMetrics)
  const currentEpoch = useTrainingStore((s) => s.currentEpoch)
  const totalEpochs = useTrainingStore((s) => s.totalEpochs)
  const currentBatch = useTrainingStore((s) => s.currentBatch)
  const totalBatches = useTrainingStore((s) => s.totalBatches)
  const currentLoss = useTrainingStore((s) => s.currentLoss)
  const etaSecs = useTrainingStore((s) => s.etaSecs)
  const errorMessage = useTrainingStore((s) => s.errorMessage)
  const customDatasetInfo = useTrainingStore((s) => s.customDatasetInfo)
  const setConfig = useTrainingStore((s) => s.setConfig)
  const startTraining = useTrainingStore((s) => s.startTraining)
  const stopTraining = useTrainingStore((s) => s.stopTraining)
  const exportWeights = useTrainingStore((s) => s.exportWeights)
  const exportONNX = useTrainingStore((s) => s.exportONNX)
  const exportFull = useTrainingStore((s) => s.exportFull)
  const csvDataset = useDatasetStore((s) => s.dataset)
  const csvTarget = useDatasetStore((s) => s.targetColumn)

  const isRunning = status === 'running' || status === 'connecting'
  const hasMetrics = epochMetrics.length > 0

  const lastMetrics = epochMetrics[epochMetrics.length - 1]

  return (
    <div style={{
      height: 300,
      background: '#0d0e14',
      borderTop: '1px solid #1e1e2e',
      display: 'flex',
      flexDirection: 'column',
      flexShrink: 0,
    }}>
      {/* Header */}
      <div style={{
        height: 36,
        display: 'flex',
        alignItems: 'center',
        padding: '0 14px',
        borderBottom: '1px solid #1e1e2e',
        gap: 10,
        flexShrink: 0,
      }}>
        <StatusBadge status={status} />

        {statusMessage && (
          <span style={{ fontSize: 11, color: '#71717a' }}>{statusMessage}</span>
        )}

        {etaSecs !== null && isRunning && (
          <span style={{ fontSize: 11, color: '#52525b' }}>
            ETA {etaSecs < 60 ? `${etaSecs}s` : `${Math.round(etaSecs / 60)}m`}
          </span>
        )}

        <div style={{ flex: 1 }} />

        {isRunning && (
          <PanelBtn onClick={stopTraining} label="Stop" danger />
        )}

        {status === 'complete' && (
          <div style={{ display: 'flex', gap: 5 }}>
            <PanelBtn onClick={exportWeights} label="↓ Weights (.pt)" />
            <PanelBtn onClick={exportFull} label="↓ Full Model" />
            <PanelBtn onClick={exportONNX} label="↓ ONNX" accent />
          </div>
        )}

        <button onClick={onClose} style={{
          background: 'transparent', border: 'none',
          color: '#52525b', cursor: 'pointer', fontSize: 16, lineHeight: 1,
          borderRadius: 4, padding: '2px 4px',
        }}
          onMouseEnter={(e) => { e.currentTarget.style.color = '#e4e4e7' }}
          onMouseLeave={(e) => { e.currentTarget.style.color = '#52525b' }}
        >×</button>
      </div>

      {/* Body */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Config / error panel */}
        <div style={{
          width: 280,
          borderRight: '1px solid #1e1e2e',
          padding: '12px 14px',
          overflowY: 'auto',
          flexShrink: 0,
        }}>
          {status === 'error' ? (
            <ErrorBlock message={errorMessage ?? 'Unknown error'} />
          ) : (
            <ConfigForm
              config={config}
              onChange={setConfig}
              onStart={startTraining}
              disabled={isRunning}
              csvDatasetName={csvDataset?.name ?? null}
              csvTarget={csvTarget}
              customDatasetInfo={customDatasetInfo}
            />
          )}
        </div>

        {/* Progress + charts */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '10px 14px', gap: 10 }}>

          {/* Progress bars */}
          {(isRunning || hasMetrics) && totalEpochs > 0 && (
            <div style={{ flexShrink: 0 }}>
              <ProgressRow
                label={`Epoch ${currentEpoch} / ${totalEpochs}`}
                value={currentEpoch / totalEpochs}
                color="#8b5cf6"
              />
              {isRunning && totalBatches > 0 && (
                <ProgressRow
                  label={`Batch ${currentBatch} / ${totalBatches}${currentLoss !== null ? ` · loss ${currentLoss.toFixed(4)}` : ''}`}
                  value={currentBatch / totalBatches}
                  color="#6366f1"
                />
              )}
              {lastMetrics && (
                <div style={{ display: 'flex', gap: 14, marginTop: 4, flexWrap: 'wrap' }}>
                  <MetricChip label="Train Loss" value={lastMetrics.trainLoss.toFixed(4)} />
                  <MetricChip label="Val Loss" value={lastMetrics.valLoss.toFixed(4)} />
                  <MetricChip label="Val Acc" value={`${(lastMetrics.valAccuracy * 100).toFixed(1)}%`} accent />
                  {lastMetrics.currentLR !== undefined && config.scheduler !== 'none' && (
                    <MetricChip label="LR" value={lastMetrics.currentLR.toExponential(2)} />
                  )}
                </div>
              )}
            </div>
          )}

          {/* Charts */}
          {hasMetrics ? (
            <div style={{ flex: 1, display: 'flex', gap: 12, minHeight: 0 }}>
              <ChartPanel
                title="Loss"
                data={epochMetrics}
                lines={[
                  { key: 'trainLoss' as keyof EpochMetrics, color: '#8b5cf6', label: 'Train' },
                  { key: 'valLoss' as keyof EpochMetrics, color: '#6366f1', label: 'Val' },
                ]}
              />
              <ChartPanel
                title="Accuracy"
                data={epochMetrics}
                lines={[{ key: 'valAccuracy' as keyof EpochMetrics, color: '#10b981', label: 'Val' }]}
                percent
              />
            </div>
          ) : status === 'idle' || status === 'stopped' || status === 'complete' ? (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <p style={{ fontSize: 12, color: '#3f3f46', textAlign: 'center', margin: 0 }}>
                {status === 'complete' ? 'Training complete.' : 'Configure and start training.'}
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

// ── Config form ───────────────────────────────────────────────────────────────

function ConfigForm({
  config,
  onChange,
  onStart,
  disabled,
  csvDatasetName,
  csvTarget,
  customDatasetInfo,
}: {
  config: TrainingConfig
  onChange: (patch: Partial<TrainingConfig>) => void
  onStart: () => void
  disabled: boolean
  csvDatasetName: string | null
  csvTarget: string | null
  customDatasetInfo: import('../types/training').CustomDatasetPayload | null
}) {
  const datasetOptions = [
    { value: 'mnist', label: 'MNIST' },
    { value: 'fashion_mnist', label: 'Fashion-MNIST' },
    { value: 'cifar10', label: 'CIFAR-10' },
    ...(csvDatasetName
      ? [{ value: 'custom', label: `CSV: ${csvDatasetName}` }]
      : [{ value: 'custom', label: 'CSV (no dataset loaded)' }]),
  ]

  const isCustom = config.dataset === 'custom'
  const customReady = isCustom && !!csvDatasetName && !!csvTarget

  return (
    <div>
      <CfgSelect
        label="Dataset"
        value={config.dataset}
        onChange={(v) => onChange({ dataset: v as TrainingConfig['dataset'] })}
        options={datasetOptions}
        disabled={disabled}
      />

      {/* Custom dataset info */}
      {isCustom && !disabled && (
        <div style={{
          marginBottom: 9, padding: '7px 9px',
          background: customReady ? 'rgba(139,92,246,0.06)' : 'rgba(239,68,68,0.06)',
          border: `1px solid ${customReady ? '#7c3aed30' : '#ef444430'}`,
          borderRadius: 6, fontSize: 10, lineHeight: 1.7,
        }}>
          {customReady ? (
            <>
              <div style={{ color: '#a78bfa', fontWeight: 600, marginBottom: 2 }}>{csvDatasetName}</div>
              <div style={{ color: '#71717a' }}>
                Target: <span style={{ color: '#e4e4e7' }}>{csvTarget}</span>
              </div>
              {customDatasetInfo && (
                <>
                  <div style={{ color: '#71717a' }}>
                    Features: <span style={{ color: '#e4e4e7' }}>{customDatasetInfo.featureCount}</span>
                    {' · '}Classes: <span style={{ color: '#e4e4e7' }}>{customDatasetInfo.classCount}</span>
                  </div>
                  <div style={{ color: '#71717a' }}>
                    Train: <span style={{ color: '#e4e4e7' }}>{customDatasetInfo.trainSamples}</span>
                    {' · '}Val: <span style={{ color: '#e4e4e7' }}>{customDatasetInfo.valSamples}</span>
                  </div>
                  <div style={{ color: '#52525b', marginTop: 2 }}>
                    Set Input node: channels={customDatasetInfo.featureCount}, H=1, W=1
                  </div>
                </>
              )}
            </>
          ) : (
            <span style={{ color: '#f87171' }}>
              Load a dataset in the Dataset tab and select a target column.
            </span>
          )}
        </div>
      )}
      <CfgNumber label="Epochs" value={config.epochs} onChange={(v) => onChange({ epochs: v })} min={1} disabled={disabled} />
      <CfgNumber label="Batch Size" value={config.batchSize} onChange={(v) => onChange({ batchSize: v })} min={1} disabled={disabled} />
      <CfgNumber label="Learning Rate" value={config.learningRate} onChange={(v) => onChange({ learningRate: v })} step={0.0001} disabled={disabled} />
      <CfgSelect
        label="Optimizer"
        value={config.optimizer}
        onChange={(v) => onChange({ optimizer: v as TrainingConfig['optimizer'] })}
        options={[
          { value: 'adam', label: 'Adam' },
          { value: 'sgd', label: 'SGD' },
          { value: 'rmsprop', label: 'RMSprop' },
        ]}
        disabled={disabled}
      />

      {/* LR Scheduler */}
      <CfgSelect
        label="LR Scheduler"
        value={config.scheduler}
        onChange={(v) => onChange({ scheduler: v as SchedulerType })}
        options={[
          { value: 'none', label: 'None' },
          { value: 'step', label: 'StepLR' },
          { value: 'cosine', label: 'CosineAnnealing' },
          { value: 'plateau', label: 'ReduceOnPlateau' },
          { value: 'exponential', label: 'ExponentialLR' },
        ]}
        disabled={disabled}
      />

      {/* Contextual scheduler params */}
      {config.scheduler === 'step' && (
        <>
          <CfgNumber label="Step Size (epochs)" value={config.schedulerStepSize} onChange={(v) => onChange({ schedulerStepSize: v })} min={1} disabled={disabled} />
          <CfgNumber label="Gamma" value={config.schedulerGamma} onChange={(v) => onChange({ schedulerGamma: v })} step={0.05} min={0} disabled={disabled} />
        </>
      )}
      {config.scheduler === 'cosine' && (
        <CfgNumber label="T_max (epochs)" value={config.schedulerTMax} onChange={(v) => onChange({ schedulerTMax: v })} min={1} disabled={disabled} />
      )}
      {config.scheduler === 'exponential' && (
        <CfgNumber label="Gamma" value={config.schedulerGamma} onChange={(v) => onChange({ schedulerGamma: v })} step={0.05} min={0} disabled={disabled} />
      )}

      <button
        onClick={onStart}
        disabled={disabled || (isCustom && !customReady)}
        style={{
          width: '100%',
          marginTop: 12,
          padding: '7px 0',
          borderRadius: 6,
          border: '1px solid #7c3aed',
          background: (disabled || (isCustom && !customReady)) ? '#1a1a2e' : '#7c3aed1a',
          color: (disabled || (isCustom && !customReady)) ? '#3f3f46' : '#a78bfa',
          fontSize: 12,
          fontWeight: 600,
          cursor: (disabled || (isCustom && !customReady)) ? 'not-allowed' : 'pointer',
          transition: 'background 0.12s',
        }}
        onMouseEnter={(e) => { if (!disabled && !isCustom || customReady) e.currentTarget.style.background = '#7c3aed33' }}
        onMouseLeave={(e) => { e.currentTarget.style.background = (disabled || (isCustom && !customReady)) ? '#1a1a2e' : '#7c3aed1a' }}
      >
        Start Training
      </button>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { color: string; bg: string; label: string }> = {
    idle: { color: '#52525b', bg: 'transparent', label: 'Idle' },
    connecting: { color: '#fbbf24', bg: 'rgba(245,158,11,0.1)', label: 'Connecting…' },
    running: { color: '#34d399', bg: 'rgba(16,185,129,0.1)', label: 'Training' },
    stopped: { color: '#a1a1aa', bg: 'rgba(161,161,170,0.1)', label: 'Stopped' },
    complete: { color: '#818cf8', bg: 'rgba(129,140,248,0.1)', label: 'Complete' },
    error: { color: '#f87171', bg: 'rgba(239,68,68,0.1)', label: 'Error' },
  }
  const c = cfg[status] ?? cfg.idle

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 5,
      padding: '2px 8px', borderRadius: 4, background: c.bg,
    }}>
      {status === 'running' && (
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#34d399', animation: 'pulse 1.5s ease-in-out infinite', flexShrink: 0 }} />
      )}
      <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: c.color }}>
        {c.label}
      </span>
    </div>
  )
}

function ProgressRow({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 10, color: '#71717a', marginBottom: 3 }}>{label}</div>
      <div style={{ height: 4, background: '#27272a', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${Math.min(value * 100, 100)}%`,
          background: color,
          borderRadius: 2,
          transition: 'width 0.3s ease',
        }} />
      </div>
    </div>
  )
}

function MetricChip({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span style={{ fontSize: 9, color: '#52525b', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 600, color: accent ? '#34d399' : '#e4e4e7', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

function ChartPanel({
  title,
  data,
  lines,
  percent,
}: {
  title: string
  data: EpochMetrics[]
  lines: { key: keyof EpochMetrics; color: string; label: string }[]
  percent?: boolean
}) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: '#52525b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
        {title}
      </div>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={data} margin={{ top: 2, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2030" />
          <XAxis dataKey="epoch" stroke="#3f3f46" tick={{ fontSize: 9, fill: '#52525b' }} />
          <YAxis
            stroke="#3f3f46"
            tick={{ fontSize: 9, fill: '#52525b' }}
            tickFormatter={percent ? (v: number) => `${(v * 100).toFixed(0)}%` : undefined}
          />
          <Tooltip
            contentStyle={{ background: '#18181b', border: '1px solid #27272a', fontSize: 11, borderRadius: 6 }}
            labelStyle={{ color: '#71717a' }}
            formatter={percent ? (v: unknown) => [`${((v as number) * 100).toFixed(1)}%`] : undefined}
          />
          {lines.length > 1 && <Legend wrapperStyle={{ fontSize: 9, color: '#71717a' }} />}
          {lines.map((l) => (
            <Line
              key={l.key}
              type="monotone"
              dataKey={l.key}
              stroke={l.color}
              strokeWidth={1.5}
              dot={false}
              name={l.label}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function ErrorBlock({ message }: { message: string }) {
  return (
    <div style={{ padding: 10, background: 'rgba(239,68,68,0.08)', border: '1px solid #ef444420', borderRadius: 6 }}>
      <p style={{ fontSize: 12, color: '#fca5a5', margin: 0, lineHeight: 1.5 }}>{message}</p>
    </div>
  )
}

function PanelBtn({ onClick, label, danger, accent }: { onClick: () => void; label: string; danger?: boolean; accent?: boolean }) {
  return (
    <button onClick={onClick} style={{
      padding: '3px 10px', borderRadius: 5,
      border: `1px solid ${danger ? '#7f1d1d' : accent ? '#7c3aed' : '#27272a'}`,
      background: danger ? 'rgba(239,68,68,0.08)' : accent ? 'rgba(124,58,237,0.12)' : 'transparent',
      color: danger ? '#f87171' : accent ? '#a78bfa' : '#71717a',
      fontSize: 11, fontWeight: 500, cursor: 'pointer',
    }}>
      {label}
    </button>
  )
}

function CfgSelect({
  label, value, onChange, options, disabled,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
  disabled?: boolean
}) {
  return (
    <div style={{ marginBottom: 9 }}>
      <div style={{ fontSize: 10, color: '#71717a', marginBottom: 3, fontWeight: 500 }}>{label}</div>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%', background: '#18181b', border: '1px solid #27272a',
          borderRadius: 5, color: disabled ? '#52525b' : '#e4e4e7',
          fontSize: 11, padding: '4px 7px', outline: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      >
        {options.map((o) => <option key={o.value} value={o.value} style={{ background: '#18181b' }}>{o.label}</option>)}
      </select>
    </div>
  )
}

function CfgNumber({
  label, value, onChange, min, step, disabled,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  step?: number
  disabled?: boolean
}) {
  return (
    <div style={{ marginBottom: 9 }}>
      <div style={{ fontSize: 10, color: '#71717a', marginBottom: 3, fontWeight: 500 }}>{label}</div>
      <input
        type="number"
        value={value}
        disabled={disabled}
        min={min}
        step={step ?? 1}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          width: '100%', background: '#18181b', border: '1px solid #27272a',
          borderRadius: 5, color: disabled ? '#52525b' : '#e4e4e7',
          fontSize: 11, padding: '4px 7px', outline: 'none',
          cursor: disabled ? 'not-allowed' : 'text',
          boxSizing: 'border-box',
        }}
      />
    </div>
  )
}
