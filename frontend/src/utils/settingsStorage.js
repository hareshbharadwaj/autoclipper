const SECRET_SETTING_KEYS = new Set([
  'groqKey',
  'geminiKey',
  'youtubeApiKey',
  'youtubeCookies',
])

// Session state fetched from the server that must never be written to
// localStorage (it is refreshed from /api/auth/settings on every load).
const NON_PERSISTED_SETTING_KEYS = new Set(['undecryptableFields'])

/** Rehydrate local UI preferences without ever accepting persisted credentials. */
export function mergeSafeSettings(defaults, persisted) {
  const settings = { ...defaults, ...(persisted || {}) }
  for (const key of SECRET_SETTING_KEYS) settings[key] = ''
  for (const key of NON_PERSISTED_SETTING_KEYS) settings[key] = []
  return settings
}

/** Return only non-secret preferences suitable for browser-local persistence. */
export function settingsForLocalStorage(settings) {
  return Object.fromEntries(
    Object.entries(settings || {}).filter(
      ([key]) => !SECRET_SETTING_KEYS.has(key) && !NON_PERSISTED_SETTING_KEYS.has(key),
    ),
  )
}

/**
 * Build the PUT /api/auth/settings payload. Blank secret fields are OMITTED
 * (never sent as empty strings): an untouched blank form after login must not
 * erase the account's stored credentials on the server.
 */
export function settingsSavePayload(settings) {
  const s = settings || {}
  const payload = {
    gemini_model: s.geminiModel || '',
    whisper_model: s.whisperModel || '',
  }
  if ((s.groqKey || '').trim()) payload.groq_key = s.groqKey.trim()
  if ((s.geminiKey || '').trim()) payload.gemini_key = s.geminiKey.trim()
  if ((s.youtubeApiKey || '').trim()) payload.youtube_api_key = s.youtubeApiKey.trim()
  if ((s.youtubeCookies || '').trim()) payload.youtube_cookies = s.youtubeCookies.trim()
  return payload
}

/**
 * Normalize the server settings response (snake_case) into the camelCase
 * shape the UI uses. Also surfaces `undecryptableFields`: credentials stored
 * under a previous encryption key that the user must save again.
 */
export function normalizeServerSettings(data) {
  const s = data || {}
  return {
    groqKey: s.groq_key ?? s.groqKey ?? '',
    geminiKey: s.gemini_key ?? s.geminiKey ?? '',
    youtubeApiKey: s.youtube_api_key ?? s.youtubeApiKey ?? '',
    geminiModel: s.gemini_model ?? s.geminiModel ?? '',
    whisperModel: s.whisper_model ?? s.whisperModel ?? '',
    youtubeCookies: s.youtube_cookies ?? s.youtubeCookies ?? '',
    undecryptableFields: Array.isArray(s.undecryptable_fields ?? s.undecryptableFields)
      ? (s.undecryptable_fields ?? s.undecryptableFields)
      : [],
  }
}
