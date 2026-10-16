import { useState, useEffect, useRef } from 'react'
import { Sparkles, Loader2, X, Wand2 } from 'lucide-react'
import { enhanceClip, enhanceDraft, enhancePreview, libraryFileUrl, cacheFileUrl } from '../api.js'

export default function EnhanceModal({ name, clip, hook, onClose, onDone }) {
  const [hookText, setHookText] = useState(hook || '')
  const [hookPosition, setHookPosition] = useState('center')
  const [hookStyle, setHookStyle] = useState('classic')
  const [hookHold, setHookHold] = useState(5.0)
  const [videoPreviewUrl, setVideoPreviewUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const debounceRef = useRef(null)


  // Auto video preview (left pane) whenever the hook changes.
  useEffect(() => {
    if (!hookText.trim()) {
      setVideoPreviewUrl('')
      return
    }
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await enhancePreview({
          name,
          filename: clip.file,
          hook_text: hookText,
          hook_position: hookPosition,
          hook_size: 'M',
          hook_style: hookStyle,
          hook_hold: hookHold,
        })
        console.log('preview resp', res)
        // unique per hook hash, so no stale cache
        setVideoPreviewUrl(cacheFileUrl(res.filename))
      } catch (e) {
        console.error('preview error', e)
        setError(`Preview gagal: ${e.message}`)
      }
    }, 600)
    return () => debounceRef.current && clearTimeout(debounceRef.current)
  }, [hookText, hookPosition, hookStyle, hookHold, name, clip.file])

  async function handleDraftHook() {
    setError('')
    setBusy(true)
    try {
      const res = await enhanceDraft({ name, filename: clip.file })
      if (res.hook) setHookText(res.hook)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleBurn() {
    setError('')
    if (!hookText.trim())
      return setError('Isi teks hook terlebih dahulu.')
    setBusy(true)
    try {
      const res = await enhanceClip({
        name,
        filename: clip.file,
        hook_text: hookText,
        hook_position: hookPosition,
        hook_size: 'M',
        hook_style: hookStyle,
        hook_hold: hookHold,
      })
      setResult(res)
      if (onDone) onDone(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4">
      <div className="bg-surface border border-border rounded-2xl w-full max-w-4xl max-h-[92vh] overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Wand2 size={18} className="text-primary" /> Enhance Clip
          </h3>
          <button onClick={onClose} className="p-1 hover:bg-white/10 rounded-full">
            <X size={18} className="text-zinc-400" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left: live video preview */}
          <div className="bg-black rounded-xl overflow-hidden flex items-center justify-center min-h-[320px]">
            {videoPreviewUrl ? (
              <video key={videoPreviewUrl} src={videoPreviewUrl} controls className="w-full" />
            ) : (
              <div className="text-center text-zinc-500 text-sm px-4">
                Preview hook akan muncul di sini saat Anda mengetik teks.
              </div>
            )}
          </div>

          {/* Right: controls */}
          <div className="space-y-5">
            {/* Hook */}
            <section>
              <div className="flex items-center gap-2 mb-2">
                <Sparkles size={15} className="text-amber-400" />
                <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Viral Hook</span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-zinc-500">Teks hook (bisa diedit)</span>
                <button
                  onClick={handleDraftHook}
                  disabled={busy}
                  className="text-xs px-2 py-1 rounded-lg bg-primary/15 text-primary hover:bg-primary/25 disabled:opacity-50"
                >
                  {busy ? '…' : 'Generate Hook (AI)'}
                </button>
              </div>
              <textarea
                value={hookText}
                onChange={(e) => setHookText(e.target.value)}
                rows={2}
                placeholder="Klik 'Generate Hook (AI)' atau ketik manual…"
                className="w-full rounded-lg bg-background border border-border px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-primary resize-y"
              />
                <div className="mt-2">
                <label className="text-xs text-zinc-400">Posisi</label>
                <select
                  value={hookPosition}
                  onChange={(e) => setHookPosition(e.target.value)}
                  className="mt-1 w-full rounded-lg bg-background border border-border px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-primary"
                >
                  <option value="top">Atas</option>
                  <option value="center">Tengah</option>
                  <option value="bottom">Bawah</option>
                </select>
              </div>
              <div className="mt-2">
                <label className="text-xs text-zinc-400">Gaya (OpenShorts)</label>
                <select
                  value={hookStyle}
                  onChange={(e) => setHookStyle(e.target.value)}
                  className="mt-1 w-full rounded-lg bg-background border border-border px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-primary"
                >
                  <option value="classic">Classic (putih)</option>
                  <option value="dark">Dark</option>
                  <option value="yellow">Yellow</option>
                  <option value="red">Red</option>
                  <option value="outline">Outline (tanpa kotak)</option>
                  <option value="outline_yellow">Outline Kuning</option>
                </select>
              </div>
              <div className="mt-2 grid grid-cols-1 gap-3">
                <div>
                  <label className="text-xs text-zinc-400">Durasi hook (detik)</label>
                  <input
                    type="number" min="1" max="30" step="0.5"
                    value={hookHold}
                    onChange={(e) => setHookHold(parseFloat(e.target.value) || 5.0)}
                    className="mt-1 w-full rounded-lg bg-background border border-border px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-primary"
                  />
                </div>
              </div>
            </section>

          </div>
        </div>

        {error && (
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 text-red-300 px-3 py-2 text-sm mt-4">
            {error}
          </div>
        )}

        {result ? (
          <div className="rounded-lg border border-green-500/40 bg-green-500/10 px-3 py-3 mt-4">
            <p className="text-sm text-green-300 mb-2">Berhasil! Klip enhanced:</p>
            <video
              src={libraryFileUrl(name, result.filename)}
              controls
              className="w-full rounded-lg bg-black mb-3"
            />
            {result.title && (
              <div className="grid grid-cols-1 gap-2 text-xs text-zinc-300 border-t border-green-500/20 pt-2 mt-1">
                <div><span className="text-zinc-500">Judul:</span> {result.title}</div>
                {result.hook && <div><span className="text-zinc-500">Hook:</span> {result.hook}</div>}
                {result.description && <div><span className="text-zinc-500">Deskripsi:</span> {result.description}</div>}
              </div>
            )}
            <div className="flex gap-3 mt-2">
              <a
                href={libraryFileUrl(name, result.filename)}
                download={result.filename}
                className="text-xs text-primary hover:underline"
              >
                Download
              </a>
              <button
                onClick={onClose}
                className="ml-auto px-3 py-1.5 rounded-lg bg-primary/20 text-primary text-xs hover:bg-primary/30"
              >
                Selesai
              </button>
            </div>
          </div>
        ) : (
          <div className="flex justify-end gap-3 mt-5">
            <button
              onClick={onClose}
              className="px-4 py-2.5 rounded-lg bg-white/5 text-zinc-300 text-sm hover:bg-white/10"
            >
              Batal
            </button>
            <button
              onClick={handleBurn}
              disabled={busy}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary hover:bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
            >
              {busy ? <Loader2 size={15} className="animate-spin" /> : <Wand2 size={15} />}
              {busy ? 'Memproses…' : 'Burn Enhance'}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
