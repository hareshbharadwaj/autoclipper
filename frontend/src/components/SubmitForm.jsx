import { useRef, useState } from 'react'
import {
  Link2, Upload, FileVideo, X, Sparkles, Scissors, LayoutGrid,
  Wand2, Loader2, ArrowLeft, RotateCcw, Clock, Type, Check, Info,
} from 'lucide-react'
import GamingLayoutModal from './GamingLayoutModal.jsx'
import { uploadFile, prepareGaming, getStatus, analyzeVideo, getAnalysis } from '../api.js'
import { normalizeAnalysisClipCount } from '../utils/clipCount.js'

const CONTENT_TYPES = [
  { id: 'general', label: 'General / Other' },
  { id: 'podcast', label: 'Podcast' },
  { id: 'gaming', label: 'Gaming' },
  { id: 'tutorial', label: 'Tutorial / How-To' },
  { id: 'irl', label: 'IRL / Live Stream' },
]

const ASPECT_OPTIONS = [
  { id: '9:16', label: '9:16', sub: 'Portrait', blurb: 'Reels · TikTok · Shorts' },
  { id: '1:1', label: '1:1', sub: 'Square', blurb: 'Feed posts' },
  { id: '16:9', label: '16:9', sub: 'Landscape', blurb: 'Original · no crop' },
]

function fmtTime(sec) {
  const s = Math.max(0, Math.round(Number(sec) || 0))
  const m = Math.floor(s / 60)
  return `${m}:${String(s % 60).padStart(2, '0')}`
}

export default function SubmitForm({ onSubmit, busy, settings }) {
  const [mode, setMode] = useState('url') // 'url' | 'upload'
  const [urlText, setUrlText] = useState('')
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadError, setUploadError] = useState('')
  const [dragActive, setDragActive] = useState(false)

  // Classic (skip-analysis) controls
  const [clipCount, setClipCount] = useState(settings?.clipCount ?? 8)
  const [minClip, setMinClip] = useState(settings?.minClip ?? 15)
  const [maxClip, setMaxClip] = useState(settings?.maxClip ?? 60)

  // Step 2 (analysis-first) controls
  const [analysisClipCount, setAnalysisClipCount] = useState(() =>
    normalizeAnalysisClipCount(settings?.analysisClipCount ?? 3)
  )
  const [targetDuration, setTargetDuration] = useState(40)
  const [customPrompt, setCustomPrompt] = useState('')
  const [aspectRatio, setAspectRatio] = useState('9:16')
  const [addHook, setAddHook] = useState(false)
  const [selectedAngle, setSelectedAngle] = useState(null)
  const [montageMode, setMontageMode] = useState(false)
  const [selectedAngles, setSelectedAngles] = useState([])
  const [voiceoverText, setVoiceoverText] = useState('')
  const [voiceoverEngine, setVoiceoverEngine] = useState('auto')
  const [backgroundMusicFile, setBackgroundMusicFile] = useState(null)
  const [musicVolume, setMusicVolume] = useState(18)

  // Analysis state
  const [analysis, setAnalysis] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisStatus, setAnalysisStatus] = useState('')
  const [analysisError, setAnalysisError] = useState('')

  const [contentType, setContentType] = useState('general')
  const [showGaming, setShowGaming] = useState(false)
  const [gamingReady, setGamingReady] = useState(null) // { source, layout }
  const [splitScreen, setSplitScreen] = useState(false)
  const fileInput = useRef(null)

  const resolvedSource = mode === 'upload' ? (file ? file.name : '') : urlText.trim()
  const hasSource = Boolean(gamingReady || resolvedSource)
  const canSubmit = !busy && !uploading && !analyzing && (analysis ? true : hasSource)

  function resetAll() {
    setAnalysis(null)
    setAnalysisError('')
    setAnalysisStatus('')
    setSelectedAngle(null)
    setSelectedAngles([])
    setCustomPrompt('')
    setContentType('general')
  }

  function handleFiles(files) {
    const f = files && files[0]
    if (!f) return
    setUploadError('')
    setFile(f)
    resetAll()
  }

  function onDrop(e) {
    e.preventDefault()
    setDragActive(false)
    handleFiles(e.dataTransfer.files)
  }

  // --- Step 1: analyze the whole video before cutting anything --------------
  async function runAnalysis() {
    if (!hasSource) return
    setUploadError('')
    setAnalysisError('')
    setAnalyzing(true)
    setAnalysis(null)
    setContentType('general')
    setAnalysisStatus('Uploading…')
    try {
      // Content type is intentionally omitted here: the analysis model detects
      // the source format from its transcript and sampled frames.
      const payload = { angle_count: 4, force_hd: Boolean(settings?.forceHd) }
      // Gaming preparation produces a server-side composed source. Prefer it
      // even when the original upload file remains selected in the form.
      const preparedSource = gamingReady?.source
      if (preparedSource) payload.source = preparedSource
      else if (!file) payload.source = urlText.trim()

      const started = await analyzeVideo(payload, preparedSource ? null : file, (p) => setUploadProgress(p))
      const analysisId = started?.analysis_id
      if (!analysisId) throw new Error('Backend did not return an analysis id.')

      const deadline = Date.now() + 1000 * 60 * 45 // 45 min safety cap
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 2000))
        const st = await getAnalysis(analysisId)
        setAnalysisStatus(st.status || 'working')
        if (st.status === 'done') {
          const result = st.result || {}
          setAnalysis(result)
          setContentType(CONTENT_TYPES.some((item) => item.id === result.content_type)
            ? result.content_type
            : 'general')
          // Pre-fill the creative controls from the strongest angle.
          const angles = st.result?.suggested_angles || []
          if (angles.length) applyAngle(angles[0], 0)
          const rec = st.result?.video_info?.aspect?.recommended
          if (rec) setAspectRatio(rec)
          return
        }
        if (st.status === 'error') {
          throw new Error(st.error || 'Analysis failed')
        }
      }
      throw new Error('Analysis timed out — the video may be too long.')
    } catch (e) {
      setAnalysisError(e.message || 'Analysis failed')
    } finally {
      setAnalyzing(false)
      setAnalysisStatus('')
      setUploadProgress(0)
    }
  }

  function applyAngle(angle, index) {
    setSelectedAngle(index)
    setSelectedAngles([])
    if (angle?.custom_prompt) setCustomPrompt(angle.custom_prompt)
    else if (angle?.title) setCustomPrompt(`Focus on: ${angle.title}. ${angle.description || ''}`.trim())
    if (angle?.target_duration) {
      setTargetDuration(Math.min(120, Math.max(10, Math.round(angle.target_duration))))
    }
  }

  async function onPrepareGaming({ camBox, gameBox, layout, setBusy, setStage, setStageMsg }) {
    const useUrl = mode === 'url' && urlText.trim()
    if (!file && !useUrl) {
      setUploadError('Upload a gaming recording or paste a YouTube URL first.')
      return
    }
    try {
      setBusy(true)
      let source
      if (file) {
        const up = await uploadFile(file, (p) => setUploadProgress(p))
        source = { file: up.filename }
      } else {
        source = { source: urlText.trim() }
      }
      const res = await prepareGaming({ ...source, camBox, gameBox, layout })
      const jobId = res.job_id
      if (!jobId) {
        // Synchronous fallback (older backend)
        setGamingReady({ source: res.source, layout })
        setShowGaming(false)
        return
      }
      const deadline = Date.now() + 1000 * 60 * 30 // 30 min max
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 3000))
        const st = await getStatus(jobId)
        if (setStage) setStage(st.status)
        const msg = (st.logs || []).slice(-1)[0] || ''
        if (setStageMsg) setStageMsg(msg || 'Working…')
        if (st.status === 'done') {
          setGamingReady({ source: st.result.source, layout, previewUrl: st.result.preview_url })
          setShowGaming(false)
          return
        }
        if (st.status === 'error') throw new Error(st.error || 'Gaming prepare failed')
      }
      throw new Error('Gaming prepare timed out — video may be too long/high-res')
    } catch (e) {
      setUploadError(e.message || 'Gaming prepare failed')
    } finally {
      setBusy(false)
    }
  }

  // --- Step 2: render --------------------------------------------------------
  function submit() {
    if (!canSubmit) return

    // Directed generation (one angle, a montage of 2-3 angles, or a custom prompt).
    if (analysis) {
      if (montageMode && (selectedAngles.length < 2 || selectedAngles.length > 3)) {
        setAnalysisError('Choose 2 or 3 distinct angles to build a montage.')
        return
      }
      const selectedClips = montageMode
        ? selectedAngles.map((index) => {
            const angle = angles[index]
            return { ...angle, start: angle.start_hint, end: angle.end_hint }
          })
        : undefined
      const selectedClip = !montageMode && selectedAngle != null
        ? { ...angles[selectedAngle], start: angles[selectedAngle]?.start_hint, end: angles[selectedAngle]?.end_hint }
        : undefined
      onSubmit({
        source: analysis.source,
        analysisId: analysis.analysis_id,
        selectedClip,
        clipCount: montageMode ? selectedClips.length : analysisClipCount,
        targetDuration: Math.min(120, Math.max(10, Number(targetDuration) || 40)),
        customPrompt: customPrompt.trim() || undefined,
        selectedClips,
        montage: montageMode || undefined,
        voiceoverText: voiceoverText.trim() || undefined,
        voiceoverEngine,
        backgroundMusicFile: backgroundMusicFile || undefined,
        musicVolume: musicVolume / 100,
        aspectRatio,
        addHook: addHook || undefined,
        splitScreen: splitScreen || undefined,
        contentType,
        analyzed: true,
      })
      return
    }

    // Classic flow: no analysis, let Gemini pick N moments.
    const clipOpts = {
      clipCount: clipCount || undefined,
      minClip: minClip || undefined,
      maxClip: maxClip || undefined,
      targetDuration: targetDuration || undefined,
      customPrompt: customPrompt.trim() || undefined,
      splitScreen: splitScreen || undefined,
      // Classic mode uses the saved vertical setting; the explicit aspect
      // selector belongs only to analysis-first mode.
      addHook: addHook || undefined,
    }
    if (gamingReady) {
      onSubmit({ source: gamingReady.source, ...clipOpts, contentType: 'general' })
      return
    }
    if (mode === 'upload') {
      setUploading(true)
      setUploadProgress(0)
      onSubmit({ file, ...clipOpts, contentType: 'general' }, (p) => setUploadProgress(p), () => setUploading(false))
    } else {
      onSubmit({ source: urlText.trim(), ...clipOpts, contentType: 'general' })
    }
  }

  const videoInfo = analysis?.video_info || {}
  const aspectMeta = videoInfo.aspect || null
  const chapters = analysis?.chapters || []
  const angles = analysis?.suggested_angles || []
  const topics = analysis?.key_topics || []

  const ANALYSIS_STAGE_LABEL = {
    queued: 'Queued…',
    downloading: 'Fetching video…',
    probing: 'Probing resolution & duration…',
    transcribing: 'Transcribing the full audio (Groq Whisper)…',
    sampling_visuals: 'Sampling timestamped frames…',
    analyzing: 'Reading transcript + visual context (Gemini)…',
    done: 'Done',
  }

  return (
    <div className="bg-surface border border-border rounded-2xl p-6">
      {/* Tabs */}
      <div className="flex items-center justify-between gap-4 mb-5 border-b border-border pb-3">
        <div className="flex gap-4">
          <TabButton active={mode === 'url'} onClick={() => setMode('url')} icon={Link2}>
            URL / Path
          </TabButton>
          <TabButton active={mode === 'upload'} onClick={() => setMode('upload')} icon={Upload}>
            Upload File
          </TabButton>
        </div>
        <StepBadge step={analysis ? 2 : 1} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        {/* ---------- Source ---------- */}
        {mode === 'url' ? (
          <input
            type="text"
            value={urlText}
            onChange={(e) => {
              setUrlText(e.target.value)
              resetAll()
            }}
            placeholder="YouTube URL or local video path (e.g. /videos/lecture.mp4)"
            className="w-full bg-background border border-border rounded-lg px-4 py-3 text-sm
                       placeholder-zinc-500 focus:outline-none focus:border-primary"
          />
        ) : (
          <div
            onClick={() => !file && !uploading && !analyzing && fileInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDragActive(true)
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
            className={`flex flex-col items-center justify-center gap-3 border-2 border-dashed
                        rounded-xl px-4 py-10 cursor-pointer transition-colors
                        ${file ? 'border-primary/50 bg-primary/5' : dragActive ? 'border-primary bg-primary/10' : 'border-zinc-700 hover:border-zinc-500 bg-white/5'}`}
          >
            <input
              ref={fileInput}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            {file ? (
              <div className="flex items-center gap-3 text-white">
                <FileVideo className="text-primary" size={20} />
                <span className="font-medium max-w-[280px] truncate">{file.name}</span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    setFile(null)
                    resetAll()
                  }}
                  className="p-1 hover:bg-white/10 rounded-full disabled:opacity-40"
                  disabled={uploading || analyzing}
                >
                  <X size={16} />
                </button>
              </div>
            ) : uploading ? (
              <div className="w-full max-w-sm">
                <p className="text-sm text-zinc-300 mb-2">Uploading… {uploadProgress}%</p>
                <div className="h-2 w-full bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-150"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            ) : (
              <>
                <Upload className="text-zinc-500" size={26} />
                <p className="text-sm text-zinc-400">Click to upload or drag and drop</p>
                <p className="text-xs text-zinc-600">MP4, MKV, MOV, WebM …</p>
              </>
            )}
          </div>
        )}

        {(uploadError || analysisError) && (
          <p className="mt-3 text-xs text-red-300">{uploadError || analysisError}</p>
        )}

        {/* ---------- Optional layouts (independent of content type) ---------- */}
        <details className="mt-5 border-t border-border pt-4">
            <summary className="cursor-pointer text-xs font-semibold text-zinc-300 uppercase tracking-wider">
              Special layouts <span className="font-normal normal-case text-zinc-500">(optional)</span>
            </summary>
            <div className="mt-3 space-y-3">
              <p className="text-[11px] text-zinc-500">
                Use these only when the source needs a camera/gameplay composite or a two-person split screen.
              </p>
              {!analysis && (
                <div className="rounded-lg border border-border bg-white/5 px-3 py-2.5">
                  {gamingReady ? (
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs text-zinc-200">Camera + gameplay layout ready ({gamingReady.layout})</span>
                      <button
                        type="button"
                        onClick={() => setShowGaming(true)}
                        className="text-xs text-primary hover:text-white underline"
                      >
                        Edit layout
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setShowGaming(true)}
                      disabled={!file && !(mode === 'url' && urlText.trim())}
                      className="w-full flex items-center justify-center gap-2 text-xs font-medium text-zinc-200
                                 disabled:opacity-40 hover:text-white transition-colors"
                    >
                      <LayoutGrid size={14} />
                      {file || (mode === 'url' && urlText.trim())
                        ? 'Prepare camera + gameplay layout'
                        : 'Upload or paste a URL first'}
                    </button>
                  )}
                </div>
              )}
              <label className="flex items-start gap-2 cursor-pointer select-none">
                <input
                  id="splitScreen"
                  type="checkbox"
                  checked={splitScreen}
                  onChange={(e) => setSplitScreen(e.target.checked)}
                  className="accent-cyan-400 mt-0.5"
                />
                <span className="text-xs text-zinc-300">
                  Two-person split screen <span className="text-zinc-500">(stacked top/bottom)</span>
                </span>
              </label>
            </div>
        </details>

        {/* ---------- STEP 1: analyze CTA ---------- */}
        {!analysis && (
          <>
            <div className="mt-5 border-t border-border pt-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <SliderRow label="Clips" value={clipCount} min={1} max={30} step={1} unit="" onChange={setClipCount} />
                <RangeSlider
                  label="Clip duration (sec)"
                  min={5}
                  max={120}
                  step={1}
                  low={minClip}
                  high={maxClip}
                  onChange={(lo, hi) => {
                    setMinClip(lo)
                    setMaxClip(hi)
                  }}
                />
              </div>
              <p className="mt-3 text-[11px] text-zinc-500">
                These are used only when you skip analysis. Analyzing first picks a specific moment instead of a
                random slice.
              </p>
            </div>

            <button
              type="button"
              onClick={runAnalysis}
              disabled={!hasSource || analyzing || busy}
              className="w-full mt-5 flex items-center justify-center gap-2 rounded-lg bg-gradient-to-r
                         from-primary to-violet-500 hover:opacity-90 disabled:opacity-50 text-white
                         font-semibold text-sm py-3 transition-opacity"
            >
              {analyzing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  {ANALYSIS_STAGE_LABEL[analysisStatus] || 'Analyzing…'}
                </>
              ) : (
                <>
                  <Wand2 size={16} />
                  Analyze Video First (Recommended)
                </>
              )}
            </button>
            <p className="mt-2 text-center text-[11px] text-zinc-500">
              Transcribes the whole video, samples a small set of timestamped frames, and returns a summary, chapters and video-specific clip angles. If frame analysis is unavailable, transcript analysis continues with a note.
            </p>
          </>
        )}

        {/* ---------- STEP 2: curate ---------- */}
        {analysis && (
          <div className="mt-5 border-t border-border pt-4 space-y-5">
            {/* Video overview */}
            <div className="rounded-xl border border-border bg-background/60 p-4">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex items-center gap-2">
                  <Info size={15} className="text-primary" />
                  <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                    Video Overview
                  </span>
                </div>
                <button
                  type="button"
                  onClick={resetAll}
                  className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-white"
                >
                  <RotateCcw size={12} /> New analysis
                </button>
              </div>
              <p className="text-sm text-zinc-200 leading-relaxed">{analysis.summary}</p>

              <div className="mt-4 rounded-lg border border-primary/25 bg-primary/5 p-3">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div>
                    <label htmlFor="analysisContentType" className="block text-xs font-semibold text-zinc-200">
                      Content type
                    </label>
                    <p className="mt-1 text-[11px] text-zinc-400">
                      Suggested after analysis: {CONTENT_TYPES.find((item) => item.id === analysis.content_type)?.label || 'General / Other'}.
                      {' '}Review or change it if the suggestion is off.
                    </p>
                  </div>
                  <select
                    id="analysisContentType"
                    value={contentType}
                    onChange={(e) => setContentType(e.target.value)}
                    className="w-full sm:w-52 rounded-lg bg-background border border-border px-3 py-2 text-sm text-white"
                  >
                    {CONTENT_TYPES.map((item) => (
                      <option key={item.id} value={item.id}>{item.label}</option>
                    ))}
                  </select>
                </div>
                {analysis.content_type_reason && (
                  <p className="mt-2 text-[11px] text-zinc-400">
                    Why this suggestion: {analysis.content_type_reason}
                  </p>
                )}
              </div>

              {analysis.analysis_warnings?.length > 0 && (
                <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">
                  <div className="font-semibold mb-1">Analysis notes</div>
                  <ul className="list-disc pl-4 space-y-1">
                    {analysis.analysis_warnings.map((warning, i) => (
                      <li key={`${i}-${warning}`}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {topics.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-3">
                  {topics.map((t) => (
                    <span
                      key={t}
                      className="px-2 py-0.5 rounded-full bg-primary/15 border border-primary/30 text-[11px] text-primary"
                    >
                      #{t}
                    </span>
                  ))}
                </div>
              )}

              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-3 text-[11px] text-zinc-500">
                {videoInfo.width ? (
                  <span>
                    {videoInfo.width}×{videoInfo.height} · {aspectMeta?.detected_label || 'unknown'}
                  </span>
                ) : null}
                <span className="flex items-center gap-1">
                  <Clock size={11} /> {fmtTime(videoInfo.duration || analysis.duration)}
                </span>
                {analysis.language ? <span>lang: {analysis.language}</span> : null}
              </div>
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-xl border border-border bg-white/5 p-3">
              <div>
                <label htmlFor="analysisClipCount" className="block text-xs font-semibold text-zinc-200">
                  Individual clips to generate
                </label>
                <p className="mt-1 text-[11px] text-zinc-500">
                  {montageMode
                    ? 'Montage mode uses the 2–3 selected moments and also creates a separate stitched montage.'
                    : 'Generate this many separate clips; the selected angle is included when its timestamps are valid.'}
                </p>
              </div>
              <select
                id="analysisClipCount"
                value={montageMode ? Math.max(2, selectedAngles.length) : analysisClipCount}
                disabled={montageMode}
                onChange={(e) => setAnalysisClipCount(normalizeAnalysisClipCount(e.target.value))}
                className="w-full sm:w-28 rounded-lg bg-background border border-border px-3 py-2 text-sm text-white disabled:opacity-50"
              >
                {Array.from({ length: 10 }, (_, i) => i + 1).map((count) => (
                  <option key={count} value={count}>{count}</option>
                ))}
              </select>
            </div>

            {/* Chapters */}
            {chapters.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Clock size={15} className="text-zinc-400" />
                  <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Timeline</span>
                </div>
                <div className="rounded-xl border border-border overflow-hidden divide-y divide-white/5">
                  {chapters.map((ch, i) => (
                    <div key={`${ch.start}-${i}`} className="flex items-center gap-3 px-3 py-2 text-xs">
                      <span className="font-mono text-primary shrink-0">{fmtTime(ch.start)}</span>
                      <span className="text-zinc-500 shrink-0">→</span>
                      <span className="font-mono text-zinc-400 shrink-0">{fmtTime(ch.end)}</span>
                      <span className="text-zinc-300 truncate">{ch.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Suggested angles */}
            {angles.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles size={15} className="text-zinc-400" />
                  <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                    Suggested viral angles
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  <label className="flex items-start gap-2 mb-3 rounded-lg border border-zinc-700 bg-white/5 p-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={montageMode}
                      onChange={(e) => {
                        setMontageMode(e.target.checked)
                        setSelectedAngles([])
                        setSelectedAngle(null)
                        setAnalysisError('')
                      }}
                      className="accent-cyan-400 mt-0.5"
                    />
                    <span className="text-xs text-zinc-300">
                      Build a multi-shot montage
                      <span className="block text-[11px] text-zinc-500">
                        Select 2–3 complementary moments; they will be stitched with short video and audio crossfades.
                      </span>
                      {montageMode && (
                        <span className="block text-[11px] text-cyan-300 mt-1">
                          {selectedAngles.length}/3 angles selected (choose at least 2)
                        </span>
                      )}
                    </span>
                  </label>
                  {angles.map((a, i) => {
                    const active = montageMode ? selectedAngles.includes(i) : selectedAngle === i
                    return (
                      <button
                        key={`${a.title}-${i}`}
                        type="button"
                        disabled={montageMode && (a.start_hint == null || a.end_hint == null)}
                        onClick={() => {
                          if (montageMode && (a.start_hint == null || a.end_hint == null)) return
                          if (!montageMode) {
                            applyAngle(a, i)
                            return
                          }
                          setSelectedAngles((prev) => {
                            if (prev.includes(i)) return prev.filter((n) => n !== i)
                            if (prev.length >= 3) return prev
                            return [...prev, i]
                          })
                          setSelectedAngle(null)
                          setCustomPrompt('Create a coherent montage from the selected moments. Keep the shots complementary, preserve their original order, and maintain a clear hook → body → payoff.')
                          setAnalysisError('')
                        }}
                        className={`text-left rounded-xl border p-3 transition-colors ${
                          active
                            ? 'border-primary bg-primary/10'
                            : 'border-zinc-700 hover:border-zinc-500 bg-white/5'
                        } ${montageMode && (a.start_hint == null || a.end_hint == null) ? 'opacity-40 cursor-not-allowed' : ''}`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <span className="text-sm font-semibold text-white leading-snug">{a.title}</span>
                          {active && <Check size={14} className="text-primary shrink-0 mt-0.5" />}
                        </div>
                        {a.hook && (
                          <p className="mt-1 text-[11px] text-amber-300/90 italic">“{a.hook}”</p>
                        )}
                        {a.description && (
                          <p className="mt-1 text-[11px] text-zinc-400 leading-snug line-clamp-3">{a.description}</p>
                        )}
                        <div className="flex items-center gap-3 mt-2 text-[10px] text-zinc-500">
                          <span>~{a.target_duration}s</span>
                          {a.start_hint != null && (
                            <span className="font-mono">
                              {fmtTime(a.start_hint)}–{fmtTime(a.end_hint)}
                            </span>
                          )}
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Creative focus prompt */}
            <div>
              <label
                htmlFor="customPrompt"
                className="flex items-center gap-2 mb-2 text-xs font-semibold text-zinc-300 uppercase tracking-wider"
              >
                <Type size={14} className="text-zinc-400" />
                Creative focus prompt
              </label>
              <textarea
                id="customPrompt"
                rows={3}
                value={customPrompt}
                onChange={(e) => {
                  setCustomPrompt(e.target.value)
                  setSelectedAngle(null)
                }}
                maxLength={1200}
                placeholder="e.g. Focus on the tunnel grip for round vegetables — open with the danger warning, show the hand placement, end once the slice is done."
                className="w-full resize-y rounded-lg bg-background border border-border px-3 py-2 text-sm
                           text-white placeholder-zinc-600 focus:outline-none focus:border-primary"
              />
              <p className="mt-1 text-[11px] text-zinc-500">
                Pick an angle above or write your own. The editor keeps a hook → body → clean ending story arc.
              </p>
            </div>

            {/* Aspect ratio */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Aspect ratio</span>
                {aspectMeta && (
                  <span className="text-[11px] text-emerald-300">
                    ✨ Recommended: {aspectMeta.recommended_label}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-3 gap-2">
                {ASPECT_OPTIONS.map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => setAspectRatio(opt.id)}
                    className={`rounded-xl border px-2 py-2.5 text-center transition-colors ${
                      aspectRatio === opt.id
                        ? 'border-primary bg-primary/10'
                        : 'border-zinc-700 hover:border-zinc-500 bg-white/5'
                    }`}
                  >
                    <div className="text-sm font-semibold text-white">{opt.label}</div>
                    <div className="text-[10px] text-zinc-400">{opt.sub}</div>
                    <div className="text-[10px] text-zinc-600 mt-0.5">{opt.blurb}</div>
                  </button>
                ))}
              </div>
              {aspectMeta && (
                <p className="mt-2 text-[11px] text-zinc-500">
                  Detected {aspectMeta.detected_label}. {aspectMeta.reason}
                </p>
              )}
            </div>

            {/* Optional Plan C audio layers */}
            <div className="space-y-3 rounded-xl border border-zinc-700 bg-white/5 p-4">
              <div>
                <div className="text-xs font-semibold text-zinc-200 uppercase tracking-wider">Optional voiceover & background music</div>
                <p className="mt-1 text-[11px] text-zinc-500">
                  Voiceover is generated from your script. When music and narration are both enabled, music volume is automatically ducked under speech.
                </p>
              </div>
              <div>
                <label htmlFor="voiceoverText" className="block mb-1 text-xs text-zinc-300">Voiceover script (optional)</label>
                <textarea
                  id="voiceoverText"
                  rows={2}
                  maxLength={5000}
                  value={voiceoverText}
                  onChange={(e) => setVoiceoverText(e.target.value)}
                  placeholder="Write narration for silent footage, B-roll, or an explanatory intro…"
                  className="w-full resize-y rounded-lg bg-background border border-border px-3 py-2 text-sm text-white placeholder-zinc-600"
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label htmlFor="voiceoverEngine" className="block mb-1 text-xs text-zinc-300">Voice engine</label>
                  <select
                    id="voiceoverEngine"
                    value={voiceoverEngine}
                    onChange={(e) => setVoiceoverEngine(e.target.value)}
                    className="w-full rounded-lg bg-background border border-border px-3 py-2 text-xs text-white"
                  >
                    <option value="auto">Automatic / fallback</option>
                    <option value="kokoro">Kokoro (local)</option>
                    <option value="edge">Edge TTS</option>
                    <option value="gemini">Gemini TTS</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="backgroundMusic" className="block mb-1 text-xs text-zinc-300">Background music (optional)</label>
                  <input
                    id="backgroundMusic"
                    type="file"
                    accept=".mp3,.wav,.m4a,.aac,.ogg,.flac,audio/*"
                    onChange={(e) => setBackgroundMusicFile(e.target.files?.[0] || null)}
                    className="block w-full text-[11px] text-zinc-400 file:mr-2 file:rounded-md file:border-0 file:bg-white/10 file:px-2 file:py-1.5 file:text-zinc-200"
                  />
                  {backgroundMusicFile && <p className="mt-1 text-[10px] text-zinc-500">{backgroundMusicFile.name}</p>}
                </div>
              </div>
              {backgroundMusicFile && (
                <SliderRow
                  label="Music level"
                  value={musicVolume}
                  min={5}
                  max={40}
                  step={1}
                  unit="%"
                  onChange={setMusicVolume}
                />
              )}
            </div>

            {/* Duration + overlays */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <SliderRow
                  label="Target duration (sec)"
                  value={targetDuration}
                  min={10}
                  max={120}
                  step={1}
                  unit="s"
                  onChange={setTargetDuration}
                />
                <p className="mt-1 text-[11px] text-zinc-500">
                  Hard cap 120s. The clip ends on a finished sentence, so it may land a few seconds off.
                </p>
              </div>
              <div className="space-y-2">
                <label className="flex items-start gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked
                    readOnly
                    className="accent-cyan-400 mt-0.5"
                  />
                  <span className="text-xs text-zinc-300">
                    Burn animated captions
                    <span className="block text-[11px] text-zinc-500">
                      Always on — word-level, synced to the speaker’s real voice (perfect lip sync).
                    </span>
                  </span>
                </label>
                <label className="flex items-start gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={addHook}
                    onChange={(e) => setAddHook(e.target.checked)}
                    className="accent-cyan-400 mt-0.5"
                  />
                  <span className="text-xs text-zinc-300">
                    Hook banner (first 4s)
                    <span className="block text-[11px] text-zinc-500">
                      Burns the angle’s hook text at the top of the clip.
                    </span>
                  </span>
                </label>
              </div>
            </div>

            <button
              type="button"
              onClick={resetAll}
              className="flex items-center gap-1 text-xs text-zinc-400 hover:text-white"
            >
              <ArrowLeft size={13} /> Back to source
            </button>
          </div>
        )}

        {/* ---------- Submit ---------- */}
        <button
          type="submit"
          disabled={!canSubmit}
          className="w-full mt-5 flex items-center justify-center gap-2 rounded-lg bg-primary
                     hover:bg-blue-600 disabled:opacity-50 text-white font-medium text-sm py-3
                     transition-colors"
        >
          {busy ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              {uploading ? `Uploading… ${uploadProgress}%` : 'Processing…'}
            </>
          ) : analysis ? (
            <>
              <Scissors size={16} />
              Generate Reel
            </>
          ) : (
            <>
              <Sparkles size={16} />
              Skip Analysis & Generate Clips
            </>
          )}
        </button>
      </form>

      {showGaming && (
        <GamingLayoutModal
          videoUrl={file ? URL.createObjectURL(file) : mode === 'url' ? urlText.trim() : ''}
          onClose={() => setShowGaming(false)}
          onPrepare={onPrepareGaming}
        />
      )}
    </div>
  )
}

function StepBadge({ step }) {
  return (
    <div className="flex items-center gap-1.5 text-[11px] shrink-0">
      <span
        className={`w-5 h-5 rounded-full flex items-center justify-center font-semibold ${
          step >= 1 ? 'bg-primary text-white' : 'bg-white/10 text-zinc-500'
        }`}
      >
        1
      </span>
      <span className="w-4 h-px bg-white/20" />
      <span
        className={`w-5 h-5 rounded-full flex items-center justify-center font-semibold ${
          step >= 2 ? 'bg-primary text-white' : 'bg-white/10 text-zinc-500'
        }`}
      >
        2
      </span>
      <span className="text-zinc-500 ml-1 hidden sm:inline">
        {step >= 2 ? 'Curate & generate' : 'Source & analyze'}
      </span>
    </div>
  )
}

function TabButton({ active, onClick, icon: Icon, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-2 pb-2 px-1 transition-all ${
        active ? 'text-primary border-b-2 border-primary -mb-[13px]' : 'text-zinc-400 hover:text-white'
      }`}
    >
      <Icon size={16} />
      {children}
    </button>
  )
}

function SliderRow({ label, value, min, max, step, unit, onChange }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-zinc-400">{label}</span>
        <span className="text-xs font-semibold text-zinc-200">
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value ?? min}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        className="w-full accent-primary"
      />
    </div>
  )
}

function RangeSlider({ label, min, max, step, low, high, onChange }) {
  const lo = Math.min(low, high)
  const hi = Math.max(low, high)
  const pct = (v) => ((v - min) / (max - min)) * 100
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-zinc-400">{label}</span>
        <span className="text-xs font-semibold text-zinc-200">
          {lo}s – {hi}s
        </span>
      </div>
      <div className="relative h-5">
        <div className="absolute top-1/2 left-0 right-0 -translate-y-1/2 h-1.5 rounded-full bg-white/10" />
        <div
          className="absolute top-1/2 -translate-y-1/2 h-1.5 rounded-full bg-primary"
          style={{ left: `${pct(lo)}%`, right: `${100 - pct(hi)}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={lo}
          onChange={(e) => onChange(Math.min(parseInt(e.target.value, 10), hi), hi)}
          className="range-thumb absolute inset-0 w-full appearance-none bg-transparent pointer-events-none"
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={hi}
          onChange={(e) => onChange(lo, Math.max(parseInt(e.target.value, 10), lo))}
          className="range-thumb absolute inset-0 w-full appearance-none bg-transparent pointer-events-none"
        />
      </div>
    </div>
  )
}
