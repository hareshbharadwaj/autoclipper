import test from 'node:test'
import assert from 'node:assert/strict'
import { mergeSafeSettings, settingsForLocalStorage } from '../src/utils/settingsStorage.js'

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
