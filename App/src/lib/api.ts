import { toScene, toAudioEvent, toAdGap } from './transforms'
import { useAppStore } from '@/store/appStore'
import { isDemoBuild } from './session'
import * as demo from './demoApi'
import type {
  Scene, AudioEvent, AdGap, Entity,
  PipelineScene, PipelineAudioEvent, PipelineAdGap,
} from '@/types'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8765'

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`fetch ${url} → ${res.status}`)
  return res.json() as Promise<T>
}

// Resolve a project from the live store (includes newly uploaded projects)
// so the editor works for projects created during this session.
function resolveProject(projectId: string) {
  return useAppStore.getState().projects.find((p) => p.id === projectId) ?? null
}

export async function fetchScenes(projectId: string): Promise<Scene[]> {
  const project = resolveProject(projectId)
  if (!project?.dataPath) throw new Error(`No pipeline data for project: ${projectId}`)
  const raw = await fetchJson<PipelineScene[]>(`${project.dataPath}/scenes.json`)
  // Filter zero-duration scenes — these are malformed pipeline outputs (start === end)
  return raw.filter((s) => s.end > s.start).map(toScene)
}

export async function fetchAudioEvents(projectId: string): Promise<AudioEvent[]> {
  const project = resolveProject(projectId)
  if (!project?.dataPath) throw new Error(`No pipeline data for project: ${projectId}`)
  const raw = await fetchJson<PipelineAudioEvent[]>(`${project.dataPath}/audio_events.json`)
  return raw.map(toAudioEvent)
}

export async function fetchAdGaps(projectId: string): Promise<AdGap[]> {
  const project = resolveProject(projectId)
  if (!project?.dataPath) throw new Error(`No pipeline data for project: ${projectId}`)
  const raw = await fetchJson<PipelineAdGap[]>(`${project.dataPath}/ad_placement_gaps.json`)
  return raw.map(toAdGap)
}

export async function fetchEntities(projectId: string): Promise<Entity[]> {
  const project = resolveProject(projectId)
  if (!project?.dataPath) throw new Error(`No pipeline data for project: ${projectId}`)
  return fetchJson<Entity[]>(`${project.dataPath}/entities.json`)
}

// ─── Editor write-through endpoints ───────────────────────────────────────────

export type VoiceId = 'onyx' | 'nova' | 'alloy' | 'shimmer'

export interface ScenePatch {
  ad?: string
  active?: boolean
  locked?: boolean
  voice?: VoiceId
  speed?: number
}

export async function previewTts(
  projectId: string,
  sceneNumber: number,
  text: string,
  voice: VoiceId,
  speed: number,
): Promise<Blob> {
  if (isDemoBuild()) return demo.demoPreviewTts(projectId, sceneNumber, voice)
  const res = await fetch(`${API_BASE}/api/jobs/${projectId}/tts-preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sceneId: `scene_${sceneNumber}`, text, voice, speed }),
  })
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText)
    throw new Error(`TTS preview failed (${res.status}): ${msg}`)
  }
  return res.blob()
}

export async function patchScene(
  projectId: string,
  sceneNumber: number,
  patch: ScenePatch,
): Promise<void> {
  if (isDemoBuild()) return demo.demoPatchScene(projectId, sceneNumber, patch)
  const res = await fetch(
    `${API_BASE}/api/jobs/${projectId}/scenes/scene_${sceneNumber}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    },
  )
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText)
    throw new Error(`Scene patch failed (${res.status}): ${msg}`)
  }
}

export async function patchEntity(
  projectId: string,
  characterId: string,
  name: string,
): Promise<void> {
  if (isDemoBuild()) return demo.demoPatchEntity()
  const res = await fetch(
    `${API_BASE}/api/jobs/${projectId}/entities/${characterId}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    },
  )
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText)
    throw new Error(`Entity rename failed (${res.status}): ${msg}`)
  }
}

export type ExportFmt = 'mp4' | 'mp3' | 'srt' | 'csv' | 'docx'

export interface ExportStatus {
  status: 'queued' | 'processing' | 'ready' | 'failed'
  progress: number
  stage: string
  format?: ExportFmt
  total_scenes?: number
  done?: number
  download_url?: string
  error?: string
}

export async function requestExport(
  projectId: string,
  voice: VoiceId,
  format: ExportFmt,
): Promise<{ exportId: string; format: ExportFmt }> {
  if (isDemoBuild()) return demo.demoRequestExport(projectId, voice, format)
  const res = await fetch(`${API_BASE}/api/jobs/${projectId}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ voice, format }),
  })
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText)
    throw new Error(`Export start failed (${res.status}): ${msg}`)
  }
  return res.json() as Promise<{ exportId: string; format: ExportFmt }>
}

export interface ServerSceneOverride {
  ad?: string
  active?: boolean
  locked?: boolean
  voice?: VoiceId
  speed?: number
}

export interface SmartFillResult {
  ad: string
  target_secs: number
  target_words: number
  estimated_secs: number
  tokens_used?: number
  model?: string
}

export async function smartFillScene(
  projectId: string,
  text: string,
  targetSecs: number,
): Promise<SmartFillResult> {
  if (isDemoBuild()) return demo.demoSmartFill(projectId, text, targetSecs)
  const res = await fetch(`${API_BASE}/api/jobs/${projectId}/smart-fill`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, target_secs: targetSecs }),
  })
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText)
    throw new Error(`Smart Fill failed (${res.status}): ${msg}`)
  }
  return res.json() as Promise<SmartFillResult>
}

export type ServerOverridesMap = Record<string, ServerSceneOverride>

export async function fetchOverrides(projectId: string): Promise<ServerOverridesMap> {
  if (isDemoBuild()) return demo.demoFetchOverrides(projectId)
  const res = await fetch(`${API_BASE}/api/jobs/${projectId}/overrides`)
  if (!res.ok) {
    if (res.status === 404) return {}
    throw new Error(`Fetch overrides failed: ${res.status}`)
  }
  return res.json() as Promise<ServerOverridesMap>
}

export async function pollExport(
  projectId: string,
  exportId: string,
): Promise<ExportStatus> {
  if (isDemoBuild()) return demo.demoPollExport()
  const res = await fetch(`${API_BASE}/api/jobs/${projectId}/export/${exportId}`)
  if (!res.ok) {
    throw new Error(`Export poll failed: ${res.status}`)
  }
  return res.json() as Promise<ExportStatus>
}

export function exportDownloadUrl(projectId: string, exportId: string, inline = false): string {
  if (isDemoBuild()) return demo.demoExportUrl(projectId)
  const q = inline ? '?inline=1' : ''
  return `${API_BASE}/api/jobs/${projectId}/export/${exportId}/download${q}`
}
