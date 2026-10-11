export const MIN_ANALYSIS_CLIP_COUNT = 1
export const MAX_ANALYSIS_CLIP_COUNT = 10
export const DEFAULT_ANALYSIS_CLIP_COUNT = 3

export function normalizeAnalysisClipCount(value, fallback = DEFAULT_ANALYSIS_CLIP_COUNT) {
  const parsed = Number(value)
  const fallbackValue = Number(fallback)
  const safeFallback = Number.isInteger(fallbackValue)
    ? Math.min(MAX_ANALYSIS_CLIP_COUNT, Math.max(MIN_ANALYSIS_CLIP_COUNT, fallbackValue))
    : DEFAULT_ANALYSIS_CLIP_COUNT
  if (!Number.isInteger(parsed)) return safeFallback
  return Math.min(MAX_ANALYSIS_CLIP_COUNT, Math.max(MIN_ANALYSIS_CLIP_COUNT, parsed))
}

// Keep the selected count in the same options object used to create FormData for
// /api/process. The backend remains responsible for authoritative validation.
export function withClipCountOption(options, value) {
  if (value === undefined || value === null || value === '') return options
  const count = Number(value)
  if (!Number.isInteger(count)) return options
  return { ...options, clip_count: count }
}
