import test from 'node:test'
import assert from 'node:assert/strict'
import {
  DEFAULT_ANALYSIS_CLIP_COUNT,
  normalizeAnalysisClipCount,
  withClipCountOption,
} from '../src/utils/clipCount.js'

test('analysis clip count stays within the visible 1–10 selector range', () => {
  assert.equal(normalizeAnalysisClipCount(1), 1)
  assert.equal(normalizeAnalysisClipCount(10), 10)
  assert.equal(normalizeAnalysisClipCount(0), 1)
  assert.equal(normalizeAnalysisClipCount(11), 10)
  assert.equal(normalizeAnalysisClipCount('not-a-number'), DEFAULT_ANALYSIS_CLIP_COUNT)
})

test('selected clip count is sent as the backend clip_count form field', () => {
  assert.deepEqual(withClipCountOption({ subtitles: true }, 3), {
    subtitles: true,
    clip_count: 3,
  })
  assert.deepEqual(withClipCountOption({ subtitles: true }, undefined), {
    subtitles: true,
  })
})
