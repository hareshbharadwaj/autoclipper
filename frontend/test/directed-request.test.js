import test from 'node:test'
import assert from 'node:assert/strict'
import {
  buildDirectedClipRequest,
  normalizeMontageTarget,
} from '../src/utils/directedRequest.js'

const ANGLES = [
  { title: 'Blade down', start_hint: 0, end_hint: 30 },
  { title: 'Claw grip', start_hint: 60, end_hint: 90 },
  { title: 'Tunnel method', start_hint: 120, end_hint: 150 },
]

test('montage mode never reduces the requested individual clip count', () => {
  const request = buildDirectedClipRequest({
    montageMode: true,
    individualClipCount: 8,
    montageTargetSeconds: 60,
    selectedAngles: [0, 1, 2],
    angles: ANGLES,
  })

  assert.equal(request.clipCount, 8)
  assert.equal(request.montage, true)
  assert.equal(request.selectedClips.length, 3)
  assert.equal(request.selectedClip, undefined)
})

test('montage off keeps the clip count and sends no montage fields', () => {
  const request = buildDirectedClipRequest({
    montageMode: false,
    individualClipCount: 2,
    montageTargetSeconds: 60,
    selectedAngles: [0, 1],
    selectedAngle: 1,
    angles: ANGLES,
  })

  assert.equal(request.clipCount, 2)
  assert.equal(request.montage, undefined)
  assert.equal(request.selectedClips, undefined)
  assert.equal(request.selectedClip.title, 'Claw grip')
  assert.equal(request.selectedClip.start, 60)
  assert.equal(request.montageTargetDuration, undefined)
})

test('the montage target is clamped and defaulted independently of clip count', () => {
  assert.equal(normalizeMontageTarget(60), 60)
  assert.equal(normalizeMontageTarget('45'), 45)
  assert.equal(normalizeMontageTarget(1), 10)
  assert.equal(normalizeMontageTarget(9999), 180)
  assert.equal(normalizeMontageTarget(undefined), 60)
  assert.equal(normalizeMontageTarget('nonsense'), 60)

  const request = buildDirectedClipRequest({
    montageMode: true,
    individualClipCount: 3,
    montageTargetSeconds: 9999,
    selectedAngles: [0, 1],
    angles: ANGLES,
  })
  assert.equal(request.montageTargetDuration, 180)
  assert.equal(request.clipCount, 3)
})
