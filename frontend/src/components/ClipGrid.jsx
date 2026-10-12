import { useState, useEffect } from 'react'
import { Download, Type, FileText, Bookmark, Sparkles, Youtube, Users, Facebook } from 'lucide-react'
import { fileUrl, saveToLibrary, applyHook, hookPreview, API_URL } from '../api.js'
import SubtitleEditor from './SubtitleEditor.jsx'
import YouTubeUploadModal from './YouTubeUploadModal.jsx'
import FacebookUploadModal from './FacebookUploadModal.jsx'
import ChatSplitModal from './ChatSplitModal.jsx'

export default function ClipGrid({ jobId, clips, onApply }) {
  const individualCount = clips.filter((clip) => !clip.is_montage).length
  const montageCount = clips.filter((clip) => clip.is_montage).length

  return (
    <section>
      <h2 className="text-lg font-semibold mb-3">
        {individualCount} individual clip{individualCount === 1 ? '' : 's'} generated
        {montageCount > 0 && ` + ${montageCount} montage${montageCount === 1 ? '' : 's'}`}
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {clips.map((clip) => (
          <ClipCard key={clip.index} jobId={jobId} clip={clip} onApply={onApply} />
        ))}
      </div>
    </section>
  )
}

function ClipQaBadge({ status }) {
  const styles = {
    pass: 'border-green-500/30 bg-green-500/10 text-green-300',
    warning: 'border-amber-500/30 bg-amber-500/10 text-amber-200',
    fail: 'border-red-500/30 bg-red-500/10 text-red-300',
    pending: 'border-zinc-500/30 bg-zinc-500/10 text-zinc-300',
  }
  const labels = {
    pass: 'QA passed',
    warning: 'QA passed with notes',
    fail: 'QA failed',
    pending: 'Awaiting final QA',
  }
  return (
    <span className={`inline-flex mt-2 rounded-full border px-2 py-0.5 text-[10px] font-medium ${styles[status] || styles.pending}`}>
      {labels[status] || status}
    </span>
  )
}

function CandidateReviewBadge({ status, score }) {
  const flagged = status === 'flagged_user_selected'
  const rating = Number.isFinite(Number(score)) ? ` · ${Number(score).toFixed(0)}/100` : ''
  return (
    <span className={`inline-flex mt-2 mr-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${
      flagged
        ? 'border-amber-500/30 bg-amber-500/10 text-amber-200'
        : 'border-cyan-500/30 bg-cyan-500/10 text-cyan-200'
    }`}>
      {flagged ? 'Reviewer flagged · kept by request' : 'Candidate reviewed'}{rating}
    </span>
  )
}

function ClipCard({ jobId, clip, onApply }) {
  const [editing, setEditing] = useState(false)
  const [showCaption, setShowCaption] = useState(false)
  const [saving, setSaving] = useState(false)
  const [savedMsg, setSavedMsg] = useState('')
  const [hooking, setHooking] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [facebookUploading, setFacebookUploading] = useState(false)
  const [splitting, setSplitting] = useState(false)
  const [splitInfo, setSplitInfo] = useState(null) // { name, clip } after auto-save
  const url = fileUrl(jobId, clip.file)
  const qa = clip.quality_assurance || {}
  const qaWarnings = [
    ...(clip.render_warnings || []),
    ...(qa.warnings || []),
    ...(qa.errors || []),
    ...(qa.tolerated_failures || []),
  ].filter((warning, index, items) => warning && items.indexOf(warning) === index)

  async function handleSave() {
    setSaving(true)
    setSavedMsg('')
    try {
      const res = await saveToLibrary({
        jobId,
        filename: clip.file,
        srt: clip.srt,
        clipTitle: clip.title,
        description: clip.description,
        hook: clip.hook,
      })
      setSavedMsg(`Saved to ${res.folder}`)
    } catch (e) {
      setSavedMsg(e.message)
    } finally {
      setSaving(false)
      setTimeout(() => setSavedMsg(''), 4000)
    }
  }

  async function handleSplit() {
    setSplitting(true)
    try {
      const res = await saveToLibrary({
        jobId,
        filename: clip.file,
        srt: clip.srt,
        clipTitle: clip.title,
        description: clip.description,
        hook: clip.hook,
      })
      setSplitInfo({ name: res.folder, clip: { file: res.file, title: clip.title } })
    } catch (e) {
      setSavedMsg(e.message)
      setTimeout(() => setSavedMsg(''), 4000)
    } finally {
      setSplitting(false)
    }
  }

  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden flex flex-col">
      <div className="relative bg-black">
        <video
          src={url}
          controls
          preload="metadata"
          className="w-full aspect-[9/16] object-contain bg-black"
        />
        <span className="absolute top-2 left-2 bg-black/60 text-white text-[10px] font-bold px-2 py-1 rounded-md uppercase tracking-wide">
          {clip.is_montage ? 'Montage' : `Clip ${clip.index}`}
        </span>
      </div>

      <div className="p-3 flex-1 flex flex-col">
        <div className="text-sm font-semibold text-zinc-100 line-clamp-2">
          {clip.title || `Clip ${clip.index}`}
        </div>
        <div className="text-xs text-zinc-500 mt-1">
          {clip.start?.toFixed(1)}s – {clip.end?.toFixed(1)}s
        </div>

        {clip.review_status && (
          <CandidateReviewBadge
            status={clip.review_status}
            score={clip.review_score}
          />
        )}
        {clip.qa_status && <ClipQaBadge status={clip.qa_status} />}
        {(clip.review_reason || clip.review_evidence || clip.visual_evidence || clip.review_warning || clip.timestamp_revision) && (
          <details className="mt-2 rounded-lg border border-cyan-500/20 bg-cyan-500/5 px-2.5 py-2">
            <summary className="cursor-pointer text-[11px] font-medium text-cyan-200">Candidate review evidence</summary>
            <div className="mt-2 space-y-1.5 text-[11px] text-zinc-300">
              {clip.review_warning && <p className="text-amber-200">{clip.review_warning}</p>}
              {clip.review_reason && <p><span className="text-zinc-500">Reason:</span> {clip.review_reason}</p>}
              {clip.review_evidence && <p><span className="text-zinc-500">Transcript:</span> “{clip.review_evidence}”</p>}
              {clip.visual_evidence && <p><span className="text-zinc-500">Visual:</span> {clip.visual_evidence}</p>}
              {clip.timestamp_revision && (
                <p>
                  <span className="text-zinc-500">Timing adjusted:</span>{' '}
                  {clip.timestamp_revision.from?.start?.toFixed?.(1)}–{clip.timestamp_revision.from?.end?.toFixed?.(1)}s to{' '}
                  {clip.timestamp_revision.to?.start?.toFixed?.(1)}–{clip.timestamp_revision.to?.end?.toFixed?.(1)}s
                </p>
              )}
            </div>
          </details>
        )}
        {qaWarnings.length > 0 && (
          <details className="mt-2 rounded-lg border border-amber-500/20 bg-amber-500/5 px-2.5 py-2">
            <summary className="cursor-pointer text-[11px] font-medium text-amber-200">
              Render and QA notes ({qaWarnings.length})
            </summary>
            <ul className="mt-2 list-disc space-y-1 pl-4 text-[11px] text-amber-100/80">
              {qaWarnings.map((warning, index) => <li key={`${index}-${warning}`}>{warning}</li>)}
            </ul>
            {qa.checks && (
              <div className="mt-2 grid grid-cols-2 gap-1 text-[10px] text-zinc-400">
                {Object.entries(qa.checks).map(([key, check]) => (
                  <span key={key}>
                    {key.replaceAll('_', ' ')}: {check?.ok === true ? 'OK' : check?.ok === false ? 'Review' : '—'}
                  </span>
                ))}
              </div>
            )}
          </details>
        )}

        {(clip.description || clip.hook) && (
          <button
            onClick={() => setShowCaption((v) => !v)}
            className="mt-2 text-left text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1"
          >
            <FileText size={13} /> {showCaption ? 'Hide caption' : 'View caption'}
          </button>
        )}

        {showCaption && (clip.description || clip.hook) && (
          <div className="mt-2 space-y-2">
            {clip.hook && (
              <div className="bg-black/20 rounded-lg p-2 border border-white/5">
                <div className="text-[10px] font-bold text-amber-400 mb-1 uppercase tracking-wider">Hook</div>
                <p className="text-xs text-zinc-300 break-words">{clip.hook}</p>
              </div>
            )}
            {clip.description && (
              <div className="bg-black/20 rounded-lg p-2 border border-white/5">
                <div className="text-[10px] font-bold text-cyan-400 mb-1 uppercase tracking-wider">Caption</div>
                <p className="text-xs text-zinc-300 break-words whitespace-pre-wrap">{clip.description}</p>
              </div>
            )}
          </div>
        )}

        <div className="flex gap-2 mt-3">
          <button
            onClick={() => setEditing(true)}
            className="flex-1 text-xs px-3 py-1.5 rounded-lg bg-accent/20 text-accent
                       hover:bg-accent/30 transition-colors flex items-center justify-center gap-1"
          >
            <Type size={13} /> Auto Subtitle
          </button>
          <button
            onClick={() => setHooking(true)}
            className="flex-1 text-xs px-3 py-1.5 rounded-lg bg-amber-500/15 text-amber-300
                       hover:bg-amber-500/25 transition-colors flex items-center justify-center gap-1"
          >
            <Sparkles size={13} /> Add Hook
          </button>
        </div>
        <div className="flex gap-2 mt-2">
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex-1 text-xs px-3 py-1.5 rounded-lg bg-white/5 text-zinc-300
                       hover:bg-white/10 transition-colors flex items-center justify-center gap-1 disabled:opacity-50"
          >
            <Bookmark size={13} /> {saving ? 'Saving…' : 'Save'}
          </button>
          <button
            onClick={() => setUploading(true)}
            className="flex-1 text-xs px-3 py-1.5 rounded-lg bg-red-600/15 text-red-400
                       hover:bg-red-600/25 transition-colors flex items-center justify-center gap-1"
          >
            <Youtube size={13} /> YouTube
          </button>
          <button
            onClick={() => setFacebookUploading(true)}
            className="flex-1 text-xs px-3 py-1.5 rounded-lg bg-[#1877f2]/15 text-[#1877f2]
                       hover:bg-[#1877f2]/25 transition-colors flex items-center justify-center gap-1"
          >
            <Facebook size={13} /> FB
          </button>
          <button
            onClick={handleSplit}
            disabled={splitting}
            className="flex-1 text-xs px-3 py-1.5 rounded-lg bg-cyan-500/15 text-cyan-300
                       hover:bg-cyan-500/25 transition-colors flex items-center justify-center gap-1 disabled:opacity-50"
          >
            <Users size={13} /> {splitting ? 'Saving…' : 'Split'}
          </button>
        </div>
        <a
          href={url}
          download={clip.file}
          className="mt-2 text-center text-xs px-3 py-1.5 rounded-lg bg-primary/20 text-primary
                     hover:bg-primary/30 transition-colors flex items-center justify-center gap-1"
        >
          <Download size={13} /> Download
        </a>
        {savedMsg && <div className="mt-2 text-xs text-green-300">{savedMsg}</div>}
      </div>

      {editing && (
        <SubtitleEditor jobId={jobId} clip={clip} onApply={onApply} onClose={() => setEditing(false)} />
      )}

      {hooking && (
        <HookModal
          clip={clip}
          jobId={jobId}
          filename={clip.file}
          onClose={() => setHooking(false)}
          onApply={async (text, position, fontScale, opts) => {
            const res = await applyHook(jobId, clip.file, text, position, fontScale, opts)
            onApply && onApply(clip.index, res.filename)
            setHooking(false)
          }}
        />
      )}

      {uploading && (
        <YouTubeUploadModal
          source={{
            jobId,
            filename: clip.file,
            title: clip.title,
            description: clip.description,
          }}
          onClose={() => setUploading(false)}
          onDone={() => setUploading(false)}
        />
      )}

      {facebookUploading && (
        <FacebookUploadModal
          source={{
            jobId,
            filename: clip.file,
            title: clip.title,
            description: clip.description,
          }}
          onClose={() => setFacebookUploading(false)}
          onDone={() => setFacebookUploading(false)}
        />
      )}

      {splitInfo && (
        <ChatSplitModal
          name={splitInfo.name}
          clip={splitInfo.clip}
          onClose={() => { setSplitInfo(null); setSplitting(false) }}
          onDone={() => { setSplitInfo(null); setSplitting(false) }}
        />
      )}
    </div>
  )
}

function HookModal({ clip, jobId, filename, onClose, onApply }) {
  const [text, setText] = useState(clip.hook || '')
  const [position, setPosition] = useState('top')
  const [size, setSize] = useState('M')
  const [entrance, setEntrance] = useState('fade')
  const [holdSeconds, setHoldSeconds] = useState(5)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [previewUrl, setPreviewUrl] = useState(null)

  const videoSrc = jobId && filename ? `${API_URL}/api/files/${jobId}/${filename}` : ''

  async function updatePreview() {
    if (!text.trim() || !jobId || !filename) {
      setPreviewUrl(null)
      return
    }
    try {
      const url = await hookPreview(jobId, filename, text, position, 1.0, { size })
      setPreviewUrl(url)
    } catch {
      setPreviewUrl(null)
    }
  }

  useEffect(() => {
    let active = true
    if (active) updatePreview()
    return () => { active = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, position, size])

  async function submit() {
    if (!text.trim()) {
      setError('Hook text is required')
      return
    }
    setBusy(true)
    setError('')
    try {
      await onApply(text, position, 1.0, { size, entrance, holdSeconds })
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  const posClass =
    position === 'center' ? 'items-center justify-center'
    : position === 'bottom' ? 'items-center justify-end pb-[20%]'
    : 'items-center justify-start pt-[20%]'

  const sizeStyle =
    size === 'S' ? { fontSize: '14px', maxWidth: '80%' }
    : size === 'L' ? { fontSize: '24px', maxWidth: '95%' }
    : { fontSize: '18px', maxWidth: '90%' }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-[#121214] border border-white/10 rounded-2xl w-full max-w-4xl p-6 relative flex flex-col md:flex-row gap-6 max-h-[90vh]">
        <button onClick={onClose} className="absolute top-4 right-4 text-zinc-500 hover:text-white z-10">✕</button>

        {/* Left: Preview */}
        <div className="flex-1 flex flex-col items-center justify-center bg-black rounded-lg border border-white/5 overflow-hidden relative aspect-[9/16] max-h-[600px]">
          {videoSrc ? (
            <>
              <video src={videoSrc} className="w-full h-full object-contain opacity-50" muted playsInline />
              <div className={`absolute w-full px-8 text-center pointer-events-none flex flex-col h-full ${posClass}`}>
                <div
                  className="text-black font-bold px-3 py-2 rounded-xl shadow-2xl text-center whitespace-pre-wrap"
                  style={{
                    ...sizeStyle,
                    backgroundColor: 'rgba(255, 255, 255, 0.82)',
                    fontFamily: 'Noto Serif, serif',
                    boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                    paddingTop: '10px', paddingBottom: '10px',
                    paddingLeft: '12px', paddingRight: '12px',
                  }}
                >
                  {text || 'Enter your text…'}
                </div>
              </div>
            </>
          ) : previewUrl ? (
            <img src={previewUrl} alt="hook preview" className="max-h-full object-contain" />
          ) : null}
        </div>

        {/* Right: Controls */}
        <div className="w-full md:w-80 flex flex-col">
          <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Sparkles size={16} className="text-amber-300" /> Viral Hook
          </h3>

          <div className="space-y-6 flex-1 overflow-y-auto pr-2">
            <div>
              <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 block">Text</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                className="w-full bg-black/40 border border-white/10 rounded-lg p-3 text-white placeholder-zinc-600
                           focus:outline-none focus:border-amber-400/50 resize-none font-serif"
                placeholder="Enter text that will stop the scroll…"
              />
            </div>

            <div>
              <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 block">Position</label>
              <div className="grid grid-cols-3 gap-2">
                {['top', 'center', 'bottom'].map((p) => (
                  <button
                    key={p}
                    onClick={() => setPosition(p)}
                    className={`py-2 px-1 rounded-lg text-xs font-bold capitalize transition-all border ${
                      position === p ? 'bg-white text-black border-white' : 'bg-white/5 text-zinc-400 border-white/5 hover:bg-white/10'
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 block">Size</label>
              <div className="grid grid-cols-3 gap-2">
                {['S', 'M', 'L'].map((sz) => (
                  <button
                    key={sz}
                    onClick={() => setSize(sz)}
                    className={`py-2 px-1 rounded-lg text-xs font-bold transition-all border ${
                      size === sz ? 'bg-white text-black border-white' : 'bg-white/5 text-zinc-400 border-white/5 hover:bg-white/10'
                    }`}
                  >
                    {sz === 'S' ? 'Small' : sz === 'M' ? 'Medium' : 'Large'}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 block">Entrance</label>
              <div className="grid grid-cols-2 gap-2">
                {[['fade', 'Fade'], ['none', 'None']].map(([val, label]) => (
                  <button
                    key={val}
                    onClick={() => setEntrance(val)}
                    className={`py-2 px-1 rounded-lg text-xs font-bold transition-all border ${
                      entrance === val ? 'bg-white text-black border-white' : 'bg-white/5 text-zinc-400 border-white/5 hover:bg-white/10'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">
                Duration: {holdSeconds}s
              </label>
              <input
                type="range"
                min="2"
                max="15"
                value={holdSeconds}
                onChange={(e) => setHoldSeconds(parseInt(e.target.value, 10))}
                className="w-full accent-amber-400"
              />
              <div className="flex justify-between text-[10px] text-zinc-500">
                <span>2s</span>
                <span>15s</span>
              </div>
            </div>
          </div>

          <button
            onClick={submit}
            disabled={busy || !text.trim()}
            className="w-full py-4 mt-4 bg-gradient-to-r from-amber-500 to-orange-500 text-black font-bold rounded-lg
                       hover:from-amber-400 hover:to-orange-400 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {busy ? 'Rendering…' : 'Add Hook'}
          </button>
        </div>
      </div>
    </div>
  )
}
