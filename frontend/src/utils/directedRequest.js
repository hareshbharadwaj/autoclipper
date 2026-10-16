// Payload shaping for the analysis-first ("directed") clip request.
//
// The rule this module exists to protect: the requested clip count always
// means *independent individual clips*. A montage is an additional output and
// must never reduce, replace, or stand in for one of them. Keeping the rule in
// one pure function makes it testable without a browser.

export const MIN_MONTAGE_SECONDS = 10
export const MAX_MONTAGE_SECONDS = 180
export const DEFAULT_MONTAGE_SECONDS = 60

export function normalizeMontageTarget(value, fallback = DEFAULT_MONTAGE_SECONDS) {
  const parsed = Number(value)
  const safe = Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
  return Math.min(MAX_MONTAGE_SECONDS, Math.max(MIN_MONTAGE_SECONDS, Math.round(safe)))
}

export function buildDirectedClipRequest({
  montageMode,
  individualClipCount,
  montageTargetSeconds,
  selectedAngles = [],
  selectedAngle = null,
  angles = [],
}) {
  const wantsMontage = Boolean(montageMode)
  const indexFor = (index) => angles[index] || {}
  const withHints = (index) => {
    const angle = indexFor(index)
    return { ...angle, start: angle.start_hint, end: angle.end_hint }
  }

  return {
    // Independent individual clips: montage mode never changes this number.
    clipCount: individualClipCount,
    montage: wantsMontage || undefined,
    selectedClips: wantsMontage ? selectedAngles.map(withHints) : undefined,
    selectedClip:
      !wantsMontage && selectedAngle != null ? withHints(selectedAngle) : undefined,
    montageTargetDuration: wantsMontage
      ? normalizeMontageTarget(montageTargetSeconds)
      : undefined,
  }
}
