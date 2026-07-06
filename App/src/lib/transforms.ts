import type {
  PipelineScene, Scene,
  PipelineAudioEvent, AudioEvent,
  PipelineAdGap, AdGap,
} from '@/types'

export function toScene(raw: PipelineScene, index: number): Scene {
  return {
    id: index + 1,
    sceneNumber: index + 1,
    startSecs: raw.start,
    endSecs: raw.end,
    durationSecs: raw.end - raw.start,
    text: raw.caption,
    template: raw.caption_template,
    characterIds: raw.character_ids,
    locked: raw.locked,
    needsReview: raw.needs_review,
    active: true,
  }
}

export function toAudioEvent(raw: PipelineAudioEvent, index: number): AudioEvent {
  const transcript = raw.transcript?.trim() || undefined
  // The audio classifier occasionally tags a non-speech sound (a music sting, a
  // title-card cue) as "dialogue" with no transcribed words. That is not real
  // dialogue: it should not paint the dialogue track or trip the AD-overlap
  // warning. Treat a wordless "dialogue" segment as non-spoken audio.
  const type: AudioEvent['type'] =
    raw.event_type === 'dialogue' && !transcript
      ? 'silence'
      : (raw.event_type as AudioEvent['type'])
  return {
    id: index + 1,
    type,
    startSecs: raw.start,
    endSecs: raw.end,
    durationSecs: raw.end - raw.start,
    transcript,
  }
}

export function toAdGap(raw: PipelineAdGap, index: number): AdGap {
  return {
    id: index + 1,
    startSecs: raw.start,
    endSecs: raw.end,
    durationSecs: raw.duration_seconds,
    isRecommended: raw.recommended,
  }
}
