import { create } from 'zustand'
import { v4 as uuid } from 'uuid'
import type { AppNode, AppEdge } from '../types/graph'
import { useGraphStore } from './useGraphStore'

const PROJECTS_KEY = 'oneiros-projects'
const LEGACY_AUTOSAVE_KEY = 'oneiros-autosave'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ProjectEntry {
  id: string
  name: string
  nodes: AppNode[]
  edges: AppEdge[]
  createdAt: number
  updatedAt: number
}

// ── Persistence helpers ───────────────────────────────────────────────────────

function loadProjects(): ProjectEntry[] {
  try {
    const raw = localStorage.getItem(PROJECTS_KEY)
    if (raw) return JSON.parse(raw) as ProjectEntry[]
  } catch { /* ignore */ }

  // One-time migration from the old single-project autosave
  try {
    const legacy = localStorage.getItem(LEGACY_AUTOSAVE_KEY)
    if (legacy) {
      const parsed = JSON.parse(legacy)
      if (parsed?.nodes) {
        const entry: ProjectEntry = {
          id: uuid(),
          name: 'My Project',
          nodes: parsed.nodes,
          edges: parsed.edges ?? [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        }
        persistProjects([entry])
        return [entry]
      }
    }
  } catch { /* ignore */ }

  return []
}

function persistProjects(projects: ProjectEntry[]): void {
  try {
    localStorage.setItem(PROJECTS_KEY, JSON.stringify(projects))
  } catch { /* quota exceeded */ }
}

// ── Store ─────────────────────────────────────────────────────────────────────

interface ProjectState {
  allProjects: ProjectEntry[]
  currentProjectId: string | null

  // Derived / single-project compat
  name: string
  isDirty: boolean

  // Multi-project actions
  createProject: (name?: string) => string
  openProject: (id: string) => void
  deleteProject: (id: string) => void
  duplicateProject: (id: string) => void
  renameProject: (id: string, name: string) => void
  goHome: () => void
  saveCurrentProject: () => void

  // Single-project compat (used by Topbar, CodePanel, etc.)
  setName: (name: string) => void
  markDirty: () => void
  markClean: () => void
  saveToFile: () => void
  loadFromFile: () => void
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  allProjects: loadProjects(),
  currentProjectId: null,
  name: 'Untitled Project',
  isDirty: false,

  // ── Multi-project ───────────────────────────────────────────────────────────

  createProject(name = 'Untitled Project') {
    const id = uuid()
    const now = Date.now()
    const entry: ProjectEntry = { id, name, nodes: [], edges: [], createdAt: now, updatedAt: now }
    const allProjects = [...get().allProjects, entry]
    persistProjects(allProjects)
    useGraphStore.getState().loadGraph({ nodes: [], edges: [] })
    set({ allProjects, currentProjectId: id, name, isDirty: false })
    return id
  },

  openProject(id) {
    const project = get().allProjects.find((p) => p.id === id)
    if (!project) return
    useGraphStore.getState().loadGraph({ nodes: project.nodes, edges: project.edges })
    set({ currentProjectId: id, name: project.name, isDirty: false })
  },

  deleteProject(id) {
    const allProjects = get().allProjects.filter((p) => p.id !== id)
    persistProjects(allProjects)
    const currentProjectId = get().currentProjectId === id ? null : get().currentProjectId
    set({ allProjects, currentProjectId })
  },

  duplicateProject(id) {
    const source = get().allProjects.find((p) => p.id === id)
    if (!source) return
    const now = Date.now()
    const copy: ProjectEntry = {
      ...source,
      id: uuid(),
      name: `${source.name} (copy)`,
      createdAt: now,
      updatedAt: now,
    }
    const allProjects = [...get().allProjects, copy]
    persistProjects(allProjects)
    set({ allProjects })
  },

  renameProject(id, name) {
    const allProjects = get().allProjects.map((p) =>
      p.id === id ? { ...p, name, updatedAt: Date.now() } : p,
    )
    persistProjects(allProjects)
    const nameUpdate = get().currentProjectId === id ? { name } : {}
    set({ allProjects, ...nameUpdate })
  },

  goHome() {
    get().saveCurrentProject()
    set({ currentProjectId: null })
  },

  saveCurrentProject() {
    const { currentProjectId, name } = get()
    if (!currentProjectId) return
    const { nodes, edges } = useGraphStore.getState().exportGraph()
    const now = Date.now()
    const allProjects = get().allProjects.map((p) =>
      p.id === currentProjectId ? { ...p, name, nodes, edges, updatedAt: now } : p,
    )
    persistProjects(allProjects)
    set({ allProjects, isDirty: false })
  },

  // ── Single-project compat ───────────────────────────────────────────────────

  setName(name) {
    set({ name, isDirty: true })
  },

  markDirty() {
    set({ isDirty: true })
  },

  markClean() {
    set({ isDirty: false })
  },

  saveToFile() {
    const { nodes, edges } = useGraphStore.getState().exportGraph()
    const { name } = get()
    const blob = new Blob([JSON.stringify({ version: 1, name, nodes, edges }, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name.replace(/\s+/g, '_').toLowerCase()}.oneiros.json`
    a.click()
    URL.revokeObjectURL(url)
    set({ isDirty: false })
  },

  loadFromFile() {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json,.oneiros.json'
    input.onchange = () => {
      const file = input.files?.[0]
      if (!file) return
      const reader = new FileReader()
      reader.onload = (ev) => {
        try {
          const payload = JSON.parse(ev.target?.result as string)
          if (!payload?.nodes) return
          const id = uuid()
          const name = payload.name ?? file.name.replace(/\.(oneiros\.)?json$/i, '')
          const now = Date.now()
          const entry: ProjectEntry = { id, name, nodes: payload.nodes, edges: payload.edges ?? [], createdAt: now, updatedAt: now }
          const allProjects = [...get().allProjects, entry]
          persistProjects(allProjects)
          useGraphStore.getState().loadGraph({ nodes: payload.nodes, edges: payload.edges ?? [] })
          set({ allProjects, currentProjectId: id, name, isDirty: false })
        } catch {
          console.error('Failed to parse project file')
        }
      }
      reader.readAsText(file)
    }
    input.click()
  },
}))
