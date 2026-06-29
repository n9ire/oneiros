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

const AUTOSAVE_DELAY = 1500

export default function App() {
  const currentProjectId = useProjectStore((s) => s.currentProjectId)
  const saveCurrentProject = useProjectStore((s) => s.saveCurrentProject)
  const markDirty = useProjectStore((s) => s.markDirty)
  const trainingStatus = useTrainingStore((s) => s.status)

  const [view, setView] = useState<AppView>('model')
  const [codeOpen, setCodeOpen] = useState(false)
  const [trainOpen, setTrainOpen] = useState(false)
  const [aiOpen, setAiOpen] = useState(false)
  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // Auto-open training panel when training starts
  useEffect(() => {
    if (trainingStatus === 'running' || trainingStatus === 'connecting') {
      setTrainOpen(true)
    }
  }, [trainingStatus])

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
      height: '100vh',
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
      />

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>
        {view === 'model' ? (
          <>
            <Sidebar />
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
              <main style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
                <FlowEditor />
                {aiOpen && <AIPanel onClose={() => setAiOpen(false)} />}
              </main>
              {codeOpen && <CodePanel onClose={() => setCodeOpen(false)} />}
              {trainOpen && <TrainingPanel onClose={() => setTrainOpen(false)} />}
            </div>
            <PropertyPanel />
          </>
        ) : (
          <DatasetPage />
        )}
      </div>
    </div>
  )
}
