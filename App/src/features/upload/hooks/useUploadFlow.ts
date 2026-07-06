import { useState, useEffect, useRef, useCallback } from 'react'
import * as uploadApi from '@/lib/uploadApi'
import { useAppStore } from '@/store/appStore'
import type { UploadSettings } from '@/types'

export type { PollResult } from '@/lib/uploadApi'

const MAX_FILE_BYTES = 4 * 1024 * 1024 * 1024  // 4 GB hard limit
const WARN_FILE_BYTES = 500 * 1024 * 1024       // 500 MB soft warning

export const DEFAULT_SETTINGS: UploadSettings = {
  mode: 'cheap',
  model: 'gpt-4.1',
  fps: 0.5,
  frameQuality: 'low',
  chunkSizeSecs: 120,
  audioExtraction: true,
  detailLevel: 3,
  presetStyle: 'documentary',
  language: null,
}

const PRESET_OVERRIDES: Record<'cheap' | 'normal', Partial<UploadSettings>> = {
  cheap:  { model: 'gpt-4.1', fps: 0.5, frameQuality: 'low', chunkSizeSecs: 120, audioExtraction: true },
  normal: { model: 'gpt-5.4', fps: 1,   frameQuality: 'low', chunkSizeSecs: 60,  audioExtraction: true },
}

export function estimateTokens(durationSecs: number, s: UploadSettings): number {
  const frames = Math.ceil(durationSecs * s.fps)
  const tokensPerFrame = s.frameQuality === 'high' ? 1105 : 85
  const chunks = Math.ceil(durationSecs / s.chunkSizeSecs)
  return (frames * tokensPerFrame) + (chunks * 7000)
}

export function estimateMinutes(durationSecs: number, s: UploadSettings): number {
  const base = s.model === 'gpt-5.4' ? 2.0 : 0.8
  const fpsFactor = s.fps === 8 ? 3 : s.fps === 0.5 ? 0.5 : 1
  return Math.max(1, Math.ceil((durationSecs / 60) * base * fpsFactor))
}

function getVideoDuration(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.onloadedmetadata = () => { URL.revokeObjectURL(url); resolve(video.duration) }
    video.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not read video duration')) }
    video.src = url
  })
}

interface UploadFlowState {
  step: 1 | 2 | 3 | 4 | 5
  projectName: string
  file: File | null
  fileUrl: string | null       // object URL for thumbnail preview
  fileDurationSecs: number
  fileSizeWarning: boolean
  customPrompt: string
  settings: UploadSettings
  jobId: string | null
  newProjectId: string | null
  progress: number
  stage: string
  chunksDone: number
  chunksTotal: number
  isReady: boolean
  isFailed: boolean
  failedError: string | null
  submitError: string | null
}

const INITIAL: UploadFlowState = {
  step: 1,
  projectName: '',
  file: null,
  fileUrl: null,
  fileDurationSecs: 0,
  fileSizeWarning: false,
  customPrompt: '',
  settings: DEFAULT_SETTINGS,
  jobId: null,
  newProjectId: null,
  progress: 0,
  stage: '',
  chunksDone: 0,
  chunksTotal: 0,
  isReady: false,
  isFailed: false,
  failedError: null,
  submitError: null,
}

export function useUploadFlow() {
  const [state, setState] = useState<UploadFlowState>(INITIAL)
  const jobIdRef = useRef<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const updateProject = useAppStore((s) => s.updateProject)

  const setFile = useCallback(async (file: File) => {
    if (file.size > MAX_FILE_BYTES) return

    const fileUrl = URL.createObjectURL(file)
    const fileSizeWarning = file.size > WARN_FILE_BYTES
    let fileDurationSecs = 0
    try { fileDurationSecs = await getVideoDuration(file) } catch { /* fallback 0 */ }

    setState((s) => ({ ...s, file, fileUrl, fileDurationSecs, fileSizeWarning }))
  }, [])

  const setProjectName = useCallback((projectName: string) => {
    setState((s) => ({ ...s, projectName }))
  }, [])

  const setCustomPrompt = useCallback((customPrompt: string) => {
    setState((s) => ({ ...s, customPrompt }))
  }, [])

  const setSettings = useCallback((patch: Partial<UploadSettings>) => {
    setState((s) => {
      const merged = { ...s.settings, ...patch }
      // apply preset overrides when mode changes away from custom
      if (patch.mode && patch.mode !== 'custom') {
        return { ...s, settings: { ...merged, ...PRESET_OVERRIDES[patch.mode] } }
      }
      return { ...s, settings: merged }
    })
  }, [])

  const next = useCallback(() => {
    setState((s) => ({ ...s, step: Math.min(s.step + 1, 5) as UploadFlowState['step'] }))
  }, [])

  const back = useCallback(() => {
    setState((s) => ({ ...s, step: Math.max(s.step - 1, 1) as UploadFlowState['step'] }))
  }, [])

  const submit = useCallback(async () => {
    if (!state.file) return
    try {
      const { jobId, projectId } = await uploadApi.submitJob(
        state.file,
        state.projectName || 'Untitled Project',
        state.settings,
        state.fileDurationSecs,
      )
      jobIdRef.current = jobId
      setState((s) => ({ ...s, jobId, newProjectId: projectId, step: 5, progress: 0 }))
    } catch {
      setState((s) => ({ ...s, submitError: 'Failed to start job. Please try again.' }))
    }
  }, [state.file, state.projectName, state.settings, state.fileDurationSecs])

  const cancel = useCallback(() => {
    if (state.fileUrl) URL.revokeObjectURL(state.fileUrl)
    if (intervalRef.current) clearInterval(intervalRef.current)
    setState(INITIAL)
  }, [state.fileUrl])

  // Polling — active only on step 5
  useEffect(() => {
    if (state.step !== 5 || !state.jobId) return

    intervalRef.current = setInterval(async () => {
      const jid = jobIdRef.current
      if (!jid) return

      let result: uploadApi.PollResult
      try {
        result = await uploadApi.pollStatus(jid)
      } catch (err) {
        // Network error — don't stop polling, just skip this tick
        console.warn('Poll error:', err)
        return
      }

      const { progress, status, stage, chunks_done, chunks_total, error,
              data_path, video_file, scene_count, tokens_used } = result

      setState((s) => ({
        ...s,
        progress,
        stage:       stage ?? '',
        chunksDone:  chunks_done ?? 0,
        chunksTotal: chunks_total ?? 0,
        isReady:     status === 'ready',
        isFailed:    status === 'failed',
        failedError: status === 'failed' ? (error ?? 'Pipeline failed') : null,
      }))

      if (status === 'ready' || status === 'failed') {
        if (intervalRef.current) clearInterval(intervalRef.current)
        if (status === 'ready' && state.newProjectId) {
          // Only patch fields the API actually returned — don't clobber an existing
          // videoFile with undefined if a stale server omits the field.
          const patch: Parameters<typeof updateProject>[1] = {
            status:     'ready',
            dataPath:   data_path ?? `/data/${state.newProjectId}`,
            sceneCount: scene_count,
            tokensUsed: tokens_used,
          }
          if (video_file) patch.videoFile = video_file
          updateProject(state.newProjectId, patch)
        }
        if (status === 'failed' && state.newProjectId) {
          updateProject(state.newProjectId, { status: 'failed' })
        }
      }
    }, 3000)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.step, state.jobId])

  const estimatedTokens = estimateTokens(state.fileDurationSecs, state.settings)
  const estimatedMinutes = estimateMinutes(state.fileDurationSecs, state.settings)

  return {
    state, estimatedTokens, estimatedMinutes,
    setFile, setProjectName, setCustomPrompt, setSettings,
    next, back, submit, cancel,
  }
}
