const STAGES = [
  ['queued', 'Queued'],
  ['ingesting', 'Preparing source'],
  ['transcribing', 'Transcribing video'],
  ['reviewing_candidates', 'Reviewing moments'],
  ['ranking_candidates', 'Ranking clips'],
  ['rendering', 'Rendering clips'],
  ['montage', 'Building montage'],
  ['audio', 'Mixing audio'],
  ['quality_assurance', 'Final quality checks'],
  ['persisting', 'Saving results'],
  ['completed', 'Complete'],
]

const REVIEW_LABELS = {
  reviewed: 'Independently reviewed',
  reviewing: 'Checking transcript and visual evidence',
  fallback_unreviewed: 'Review unavailable — transcript-validated timings retained',
  skipped_no_key: 'Review skipped — provider key not configured',
  not_run: 'Review not run',
}

const QA_LABELS = {
  pass: 'Passed',
  warning: 'Passed with warnings',
  fail: 'Failed',
  not_run: 'Not run',
  running: 'In progress',
}

export default function JobProgress({ job }) {
  if (!job) return null

  const progress = job.progress || job.result?.progress || {}
  const stage = progress.stage || job.status || 'queued'
  const stageIndex = STAGES.findIndex(([key]) => key === stage)
  const stageLabel = STAGES[stageIndex]?.[1] || (job.status === 'failed' ? 'Processing failed' : 'Working')
  const total = Number(progress.total)
  const current = Number(progress.current)
  const hasCounts = Number.isFinite(total) && total > 0 && Number.isFinite(current)
  const percentage = hasCounts
    ? Math.max(0, Math.min(100, Math.round((current / total) * 100)))
    : job.status === 'completed' ? 100 : job.status === 'failed' ? 0 : null
  const isWorking = job.status === 'queued' || job.status === 'processing'
  const review = job.result?.candidate_review || progress.candidate_review ||
    (stage === 'reviewing_candidates' ? { status: 'reviewing' } : null)
  const qa = job.result?.quality_assurance
  const qaStatus = qa?.status || (stage === 'quality_assurance' ? progress.qa_status || 'running' : null)
  const qaWarnings = (qa?.reports || []).reduce((count, report) => count + (report.status === 'warning' ? 1 : 0), 0)
  const qaFailed = qa?.failed_clip_count ?? (qa?.reports || []).filter((report) => report.status === 'fail').length

  return (
    <section
      className="mb-3 rounded-lg border border-border bg-surface/70 px-4 py-3"
      aria-label="Clip generation progress"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-zinc-100">{stageLabel}</div>
          <p className="mt-1 text-xs text-zinc-400" aria-live="polite">
            {progress.message || (isWorking ? 'Preparing the processing pipeline…' : 'Job finished.')}
          </p>
        </div>
        {stageIndex >= 0 && (
          <span className="shrink-0 text-[10px] uppercase tracking-wide text-zinc-500">
            Stage {stageIndex + 1} / {STAGES.length}
          </span>
        )}
      </div>

      <div
        className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10"
        role="progressbar"
        aria-label={stageLabel}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percentage ?? undefined}
        aria-valuetext={hasCounts ? `${current} of ${total}` : stageLabel}
      >
        <div
          className={`h-full rounded-full bg-primary transition-all duration-500 ${
            percentage == null ? 'w-1/3 animate-pulse' : ''
          }`}
          style={percentage == null ? undefined : { width: `${percentage}%` }}
        />
      </div>
      {hasCounts && (
        <div className="mt-1 text-right text-[10px] text-zinc-500">
          {current} of {total}
        </div>
      )}

      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        {review && (
          <div className="rounded-md bg-black/20 px-3 py-2 text-xs">
            <div className="font-medium text-zinc-200">Candidate review</div>
            <div className="mt-0.5 text-zinc-400">
              {REVIEW_LABELS[review.status] || review.status || 'Pending'}
              {review.status === 'reviewed' && (
                <span>
                  {' '}· {review.reviewed_count || 0} reviewed, {review.rejected_count || 0} rejected
                </span>
              )}
            </div>
          </div>
        )}
        {qaStatus && (
          <div className="rounded-md bg-black/20 px-3 py-2 text-xs">
            <div className="font-medium text-zinc-200">Rendered-file QA</div>
            <div className="mt-0.5 text-zinc-400">
              {QA_LABELS[qaStatus] || qaStatus}
              {qa && (
                <span>
                  {' '}· {qa.checked_clip_count ?? qa.reports?.length ?? 0} checked,
                  {' '}{qaFailed} failed, {qaWarnings} with warnings
                </span>
              )}
              {!qa && stage === 'quality_assurance' && hasCounts && (
                <span>{' '}· {current} of {total} checked</span>
              )}
              {qa?.failed_outputs?.length > 0 && (
                <ul className="mt-2 list-disc space-y-1 pl-4 text-amber-200/90">
                  {qa.failed_outputs.map((item, index) => (
                    <li key={`${item.index || index}-${item.file || ''}`}>
                      Clip {item.index || index + 1}: {(item.quality_assurance?.errors || [])[0] || 'No verified playable render was available.'}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
