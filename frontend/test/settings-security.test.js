import test from 'node:test'
import assert from 'node:assert/strict'
import { mergeSafeSettings, normalizeServerSettings, settingsForLocalStorage, settingsSavePayload } from '../src/utils/settingsStorage.js'

const defaults = {
  groqKey: '',
  geminiKey: '',
  youtubeApiKey: '',
  youtubeCookies: '',
  vertical: true,
  clipCount: 8,
}

test('legacy local settings never rehydrate provider credentials', () => {
  const settings = mergeSafeSettings(defaults, {
    groqKey: 'old-groq-secret',
    geminiKey: 'old-gemini-secret',
    youtubeApiKey: 'old-youtube-secret',
    youtubeCookies: 'old-cookie-secret',
    clipCount: 4,
  })
  assert.equal(settings.groqKey, '')
  assert.equal(settings.geminiKey, '')
  assert.equal(settings.youtubeApiKey, '')
  assert.equal(settings.youtubeCookies, '')
  assert.equal(settings.clipCount, 4)
})

test('local settings storage retains preferences but strips every secret field', () => {
  const stored = settingsForLocalStorage({
    ...defaults,
    groqKey: 'gsk-secret',
    geminiKey: 'gemini-secret',
    youtubeApiKey: 'youtube-secret',
    youtubeCookies: 'cookie-secret',
    vertical: false,
  })
  assert.deepEqual(stored, { vertical: false, clipCount: 8 })
  assert.equal(JSON.stringify(stored).includes('secret'), false)
})

test('server settings normalize to camelCase and surface undecryptable fields', () => {
  const normalized = normalizeServerSettings({
    groq_key: 'gsk_from_server',
    gemini_key: 'gemini_from_server',
    youtube_api_key: null,
    gemini_model: 'gemini-3.5-flash',
    whisper_model: 'whisper-large-v3-turbo',
    youtube_cookies: 'cookies_from_server',
    undecryptable_fields: ['groq_key', 'youtube_token'],
  })
  assert.deepEqual(normalized, {
    groqKey: 'gsk_from_server',
    geminiKey: 'gemini_from_server',
    youtubeApiKey: '',
    geminiModel: 'gemini-3.5-flash',
    whisperModel: 'whisper-large-v3-turbo',
    youtubeCookies: 'cookies_from_server',
    undecryptableFields: ['groq_key', 'youtube_token'],
  })
})

test('missing or malformed undecryptable_fields degrade to an empty list', () => {
  assert.deepEqual(normalizeServerSettings({}).undecryptableFields, [])
  assert.deepEqual(normalizeServerSettings(null).undecryptableFields, [])
  assert.deepEqual(
    normalizeServerSettings({ undecryptable_fields: 'groq_key' }).undecryptableFields,
    [],
  )
})

test('undecryptable field notices never persist to localStorage', () => {
  const settings = mergeSafeSettings(defaults, {
    clipCount: 5,
    undecryptableFields: ['groq_key'],
  })
  assert.deepEqual(settings.undecryptableFields, [])
  const stored = settingsForLocalStorage({ ...settings, undecryptableFields: ['groq_key'] })
  assert.equal('undecryptableFields' in stored, false)
  assert.deepEqual(stored, { vertical: true, clipCount: 5 })
})

test('save payload omits blank secret fields and trims provided ones', () => {
  const payload = settingsSavePayload({
    groqKey: '',
    geminiKey: '   ',
    youtubeApiKey: '  AIza_example  ',
    youtubeCookies: '',
    geminiModel: 'gemini-3.5-flash',
    whisperModel: '',
  })
  assert.deepEqual(payload, {
    gemini_model: 'gemini-3.5-flash',
    whisper_model: '',
    youtube_api_key: 'AIza_example',
  })
  assert.equal('groq_key' in payload, false)
  assert.equal('gemini_key' in payload, false)
  assert.equal('youtube_cookies' in payload, false)
})

test('save payload sends every filled secret field', () => {
  const payload = settingsSavePayload({
    groqKey: 'gsk_a',
    geminiKey: 'AIza_b',
    youtubeApiKey: 'AIza_c',
    youtubeCookies: 'cookies',
    geminiModel: 'm',
    whisperModel: 'w',
  })
  assert.deepEqual(payload, {
    groq_key: 'gsk_a',
    gemini_key: 'AIza_b',
    youtube_api_key: 'AIza_c',
    youtube_cookies: 'cookies',
    gemini_model: 'm',
    whisper_model: 'w',
  })
})
