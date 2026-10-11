const SECRET_SETTING_KEYS = new Set([
  'groqKey',
  'geminiKey',
  'youtubeApiKey',
  'youtubeCookies',
])

/** Rehydrate local UI preferences without ever accepting persisted credentials. */
export function mergeSafeSettings(defaults, persisted) {
  const settings = { ...defaults, ...(persisted || {}) }
  for (const key of SECRET_SETTING_KEYS) settings[key] = ''
  return settings
}

/** Return only non-secret preferences suitable for browser-local persistence. */
export function settingsForLocalStorage(settings) {
  return Object.fromEntries(
    Object.entries(settings || {}).filter(([key]) => !SECRET_SETTING_KEYS.has(key)),
  )
}
