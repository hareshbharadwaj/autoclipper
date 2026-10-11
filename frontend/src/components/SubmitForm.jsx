import { useRef, useState } from 'react'
import {
  Link2, Upload, FileVideo, X, Sparkles, Scissors, LayoutGrid,
  Wand2, Loader2, ArrowLeft, RotateCcw, Clock, Type, Check, Info,
} from 'lucide-react'
import GamingLayoutModal from './GamingLayoutModal.jsx'
import { uploadFile, prepareGaming, getStatus, analyzeVideo, getAnalysis } from '../api.js'

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
  const [targetDuration, setTargetDuration] = useState(40)
  const [customPrompt, setCustomPrompt] = useState('')
  const [aspectRatio, setAspectRatio] = useState('9:16')
  const [addHook, setAddHook] = useState(false)
  const [selectedAngle, setSelectedAngle] = useState(null)

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
    setCustomPrompt('')
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
    setAnalysisStatus('Uploading…')
    try {
      const payload = { content_type: contentType, angle_count: 4 }
      if (!file) payload.source = gamingReady ? gamingReady.source : urlText.trim()

      const started = await analyzeVideo(payload, file, (p) => setUploadProgress(p))
      const analysisId = started?.analysis_id
      if (!analysisId) throw new Error('Backend did not return an analysis id.')

      const deadline = Date.now() + 1000 * 60 * 45 // 45 min safety cap
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 2000))
        const st = await getAnalysis(analysisId)
        setAnalysisStatus(st.status || 'working')
        if (st.status === 'done') {
          setAnalysis(st.result || {})
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

    // Directed generation (an angle was chosen / a focus prompt was typed).
    if (analysis) {
      onSubmit({
        source: analysis.source,
        clipCount: 1,
        targetDuration: Math.min(120, Math.max(10, Number(targetDuration) || 40)),
        customPrompt: customPrompt.trim() || undefined,
        aspectRatio,
        addHook: addHook || undefined,
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
      aspectRatio,
      addHook: addHook || undefined,
    }
    if (contentType === 'gaming' && gamingReady) {
      onSubmit({ source: gamingReady.source, ...clipOpts, contentType })
      return
    }
    if (mode === 'upload') {
      setUploading(true)
      setUploadProgress(0)
      onSubmit({ file, ...clipOpts, contentType }, (p) => setUploadProgress(p), () => setUploading(false))
    } else {
      onSubmit({ source: urlText.trim(), ...clipOpts, contentType })
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
    analyzing: 'Reading the whole video (Gemini)…',
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

        {/* ---------- Content type ---------- */}
        <div className="mt-5 border-t border-border pt-4">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles size={15} className="text-zinc-400" />
            <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Content Type</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {CONTENT_TYPES.map((ct) => (
              <button
                key={ct.id}
                type="button"
                onClick={() => setContentType(ct.id)}
                className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                  contentType === ct.id
                    ? 'bg-primary/20 border-primary text-primary'
                    : 'border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500'
                }`}
              >
                {ct.label}
              </button>
            ))}
          </div>

          {contentType === 'gaming' && (
            <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2.5">
              {gamingReady ? (
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-amber-200">✓ Gaming video ready ({gamingReady.layout})</span>
                  <button
                    type="button"
                    onClick={() => setShowGaming(true)}
                    className="text-xs text-amber-300 hover:text-amber-100 underline"
                  >
                    Edit layout
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setShowGaming(true)}
                  disabled={!file && mode !== 'url'}
                  className="w-full flex items-center justify-center gap-2 text-xs font-bold text-amber-200
                             disabled:opacity-40 hover:text-amber-100 transition-colors"
                >
                  <LayoutGrid size={14} />
                  {file || (mode === 'url' && urlText.trim())
                    ? 'Setup Cam & Gameplay'
                    : 'Upload or paste a URL first'}
                </button>
              )}
            </div>
          )}

          {contentType === 'podcast' && (
            <div className="mt-3 flex items-center gap-2">
              <input
                id="splitScreen"
                type="checkbox"
                checked={splitScreen}
                onChange={(e) => setSplitScreen(e.target.checked)}
                className="accent-cyan-400"
              />
              <label htmlFor="splitScreen" className="text-xs text-zinc-300 cursor-pointer select-none">
                Split Screen <span className="text-zinc-500">(2 orang, atas/bawah)</span>
              </label>
            </div>
          )}
        </div>

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
              Transcribes the whole video and returns a summary, timestamped chapters and clip angles unique to it.
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
                  {angles.map((a, i) => {
                    const active = selectedAngle === i
                    return (
                      <button
                        key={`${a.title}-${i}`}
                        type="button"
                        onClick={() => applyAngle(a, i)}
                        className={`text-left rounded-xl border p-3 transition-colors ${
                          active
                            ? 'border-primary bg-primary/10'
                            : 'border-zinc-700 hover:border-zinc-500 bg-white/5'
                        }`}
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
