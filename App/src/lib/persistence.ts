interface SceneEdit {
  text?: string
  active?: boolean
}

interface ProjectEdits {
  scenes: Record<number, SceneEdit>
}

function key(projectId: string): string {
  return `instascribe:${projectId}:edits`
}

export function loadEdits(projectId: string): ProjectEdits {
  try {
    const raw = localStorage.getItem(key(projectId))
    if (!raw) return { scenes: {} }
    return JSON.parse(raw) as ProjectEdits
  } catch {
    return { scenes: {} }
  }
}

function saveEdits(projectId: string, edits: ProjectEdits): void {
  localStorage.setItem(key(projectId), JSON.stringify(edits))
}

export function persistSceneText(projectId: string, sceneId: number, text: string): void {
  const edits = loadEdits(projectId)
  edits.scenes[sceneId] = { ...edits.scenes[sceneId], text }
  saveEdits(projectId, edits)
}

export function persistSceneActive(projectId: string, sceneId: number, active: boolean): void {
  const edits = loadEdits(projectId)
  edits.scenes[sceneId] = { ...edits.scenes[sceneId], active }
  saveEdits(projectId, edits)
}
