import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { DEMO_USER } from '@/features/auth/constants'
import { PROJECTS } from '@/lib/projects'
import { deleteProjectOnServer, patchProjectOnServer } from '@/lib/uploadApi'
import type { User, Project } from '@/types'

interface AppState {
  currentUser: User | null
  isAuthenticated: boolean
  sidebarCollapsed: boolean
  isDemoMode: boolean
  projects: Project[]
  login: (email: string, password: string) => boolean
  logout: () => void
  addProject: (project: Project) => void
  updateProjectStatus: (id: string, status: Project['status']) => void
  updateProject: (id: string, patch: Partial<Project>) => void
  deleteProject: (id: string) => Promise<void>
  renameProject: (id: string, name: string) => Promise<void>
  toggleStar: (id: string) => Promise<void>
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      currentUser: null,
      isAuthenticated: false,
      sidebarCollapsed: false,
      isDemoMode: true,
      projects: PROJECTS,
      login: (email, password) => {
        if (email === DEMO_USER.email && password === DEMO_USER.password) {
          set({
            currentUser: { email: DEMO_USER.email, name: DEMO_USER.name, tokenBalance: 1_000_000 },
            isAuthenticated: true,
          })
          return true
        }
        return false
      },
      logout: () => set({ currentUser: null, isAuthenticated: false }),
      addProject: (project) => set((s) => ({ projects: [project, ...s.projects] })),
      updateProjectStatus: (id, status) =>
        set((s) => ({
          projects: s.projects.map((p) => (p.id === id ? { ...p, status } : p)),
        })),
      updateProject: (id, patch) =>
        set((s) => ({
          projects: s.projects.map((p) => (p.id === id ? { ...p, ...patch } : p)),
        })),
      deleteProject: async (id) => {
        await deleteProjectOnServer(id)
        set((s) => ({ projects: s.projects.filter((p) => p.id !== id) }))
      },
      renameProject: async (id, name) => {
        const trimmed = name.trim()
        if (!trimmed) return
        await patchProjectOnServer(id, { name: trimmed })
        set((s) => ({
          projects: s.projects.map((p) => (p.id === id ? { ...p, name: trimmed } : p)),
        }))
      },
      toggleStar: async (id) => {
        const project = get().projects.find((p) => p.id === id)
        if (!project) return
        const next = !project.starred
        await patchProjectOnServer(id, { starred: next })
        set((s) => ({
          projects: s.projects.map((p) => (p.id === id ? { ...p, starred: next } : p)),
        }))
      },
    }),
    {
      name: 'instascribe-app',
      partialize: (state) => ({
        currentUser: state.currentUser,
        isAuthenticated: state.isAuthenticated,
        sidebarCollapsed: state.sidebarCollapsed,
        isDemoMode: state.isDemoMode,
        projects: state.projects,
      }),
    }
  )
)
