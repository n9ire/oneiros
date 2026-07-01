import { useEffect, useRef, useState } from 'react'
import FlowEditor from './editor/FlowEditor'
import Sidebar from './components/Sidebar'
import PropertyPanel from './components/PropertyPanel'
import Topbar from './components/Topbar'
import type { AppView } from './components/Topbar'
import CodePanel from './components/CodePanel'
import TrainingPanel from './components/TrainingPanel'
import AIPanel from './components/AIPanel'
import DatasetPage from './pages/DatasetPage'
import ProjectsPage from './pages/ProjectsPage'
import { useGraphStore } from './store/useGraphStore'
import { useProjectStore } from './store/useProjectStore'
import { useTrainingStore } from './store/useTrainingStore'
import { useIsMobile } from './hooks/useBreakpoint'

const AUTOSAVE_DELAY = 1500

export default function App() {
  const currentProjectId = useProjectStore((s) => s.currentProjectId)
  const saveCurrentProject = useProjectStore((s) => s.saveCurrentProject)
  const markDirty = useProjectStore((s) => s.markDirty)
  const trainingStatus = useTrainingStore((s) => s.status)
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const isMobile = useIsMobile()

  const [view, setView] = useState<AppView>('model')
  const [codeOpen, setCodeOpen] = useState(false)
  const [trainOpen, setTrainOpen] = useState(false)
  const [aiOpen, setAiOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // Auto-open training panel when training starts
  useEffect(() => {
    if (trainingStatus === 'running' || trainingStatus === 'connecting') {
      setTrainOpen(true)
    }
  }, [trainingStatus])

  // On mobile, open inspector when a node is selected
  useEffect(() => {
    if (isMobile && view === 'model' && selectedNodeId) {
      setInspectorOpen(true)
    }
  }, [isMobile, view, selectedNodeId])

  // Close drawers when leaving model view
  useEffect(() => {
    if (view !== 'model') {
      setPaletteOpen(false)
      setInspectorOpen(false)
    }
  }, [view])

  // Auto-save to project store on graph change
  useEffect(() => {
    if (!currentProjectId) return
    const unsubscribe = useGraphStore.subscribe(() => {
      markDirty()
      clearTimeout(saveTimer.current)
      saveTimer.current = setTimeout(() => {
        saveCurrentProject()
      }, AUTOSAVE_DELAY)
    })
    return () => {
      unsubscribe()
      clearTimeout(saveTimer.current)
    }
  }, [currentProjectId, markDirty, saveCurrentProject])

  // Keyboard shortcuts
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const mod = e.metaKey || e.ctrlKey
      if (!mod) return
      if (e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        useGraphStore.getState().undo()
      }
      if ((e.key === 'z' && e.shiftKey) || e.key === 'y') {
        e.preventDefault()
        useGraphStore.getState().redo()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  // Show projects home when no project is open
  if (!currentProjectId) {
    return <ProjectsPage />
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100dvh',
      width: '100vw',
      background: '#09090b',
      overflow: 'hidden',
    }}>
      <Topbar
        view={view}
        onViewChange={setView}
        codeOpen={codeOpen}
        onToggleCode={() => setCodeOpen((o) => !o)}
        trainOpen={trainOpen}
        onToggleTrain={() => setTrainOpen((o) => !o)}
        aiOpen={aiOpen}
        onToggleAI={() => setAiOpen((o) => !o)}
        isMobile={isMobile}
        paletteOpen={paletteOpen}
        onTogglePalette={() => setPaletteOpen((o) => !o)}
        inspectorOpen={inspectorOpen}
        onToggleInspector={() => setInspectorOpen((o) => !o)}
      />

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0, position: 'relative' }}>
        {view === 'model' ? (
          <>
            {!isMobile && <Sidebar />}
            {isMobile && (
              <Sidebar mobile open={paletteOpen} onClose={() => setPaletteOpen(false)} />
            )}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
              <main style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
                <FlowEditor />
                {aiOpen && <AIPanel onClose={() => setAiOpen(false)} mobile={isMobile} />}
              </main>
              {codeOpen && <CodePanel onClose={() => setCodeOpen(false)} mobile={isMobile} />}
              {trainOpen && <TrainingPanel onClose={() => setTrainOpen(false)} mobile={isMobile} />}
            </div>
            {!isMobile && <PropertyPanel />}
            {isMobile && (
              <PropertyPanel mobile open={inspectorOpen} onClose={() => setInspectorOpen(false)} />
            )}
          </>
        ) : (
          <DatasetPage mobile={isMobile} />
        )}
      </div>
    </div>
  )
}
