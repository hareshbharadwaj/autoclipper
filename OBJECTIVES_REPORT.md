# Request-compliance work: report

Baseline: `main` @ `fded42dd8101c3b553ff58ba1bc26bfdb6c627bd` (tag `good-enough-baseline`, PR #8 merged).
Branch: `arena/c6ac8e24-autoclipper` (the dedicated feature branch for this work).
Prepared by **Arena.ai Agent Mode**. Nothing merged, nothing pushed to `main`.

---

## 1. What the manual test exposed

Your knife-safety run asked for **2 clips at 60 s** and produced Clip 1 = 30 s,
Clip 2 = 26.6 s, plus a ~56.75 s montage, with the job marked *completed* and QA
*passed*. All three observations were real defects, not user error.

### Defect A — a 60 s request silently became a 30 s request

**Root cause: montage mode took ownership of the clip count and the duration budget.**
Two places, one in each tier:

| Tier | Location (baseline) | Behaviour |
|---|---|---|
| Backend | `pipeline.py` `run()` — `if montage:` branch | `requested_individual_count = len(selected_clips)` (2), and `target_duration` was handed to `_fit_montage_candidates()` as the **montage's shared budget**, which divides it by the shot count → `desired_per_shot = 60 / 2 = 30`. Each angle then became one individual clip planned with `target_duration=None`, inheriting only its 30 s share. |
| Frontend | `SubmitForm.jsx:229` | `clipCount: montageMode ? selectedClips.length : analysisClipCount` — with `SubmitForm.jsx:586` also *disabling* the count selector in montage mode. The user could not express "2 clips" and "a montage" at the same time. |

So `target_duration` was never applied to the clips you saw; "60" only ever meant
"montage".

### Defect B — the shortfall was not reported and the job said "completed"

**Root cause: the shortfall check was skipped in montage mode, and the check
compared the render against the *plan*, not against your request.**

- `pipeline.py`: `if not montage and actual_individual_count < requested_individual_count:` — the guard made the check unreachable whenever a montage was requested. With `requested_individual_count` also rewritten to 2, there was nothing left to compare.
- `quality_assurance.verify_rendered_file()` compared the rendered file against `expected_duration` derived from the **edit plan**. A clip planned for 30 s and rendered at 30 s therefore "passed", even though the request was 60 s.
- `app.py` set `job["status"] = "completed"` whenever the pipeline returned without raising, so a shortfall had no way to reach the UI.

### Defect C — Windows `UnicodeEncodeError` in the hook overlay

**Root cause: `hooks.py` wrote emoji to the process stdout with `print()`.**
On Windows, `sys.stdout` defaults to a legacy code page (cp1252), so
`print("Hook overlay ✅ added")` raises `UnicodeEncodeError`. Two contributing
layers:

- `utils.run_command()` used `subprocess.run(..., text=True)` without an explicit
  encoding, so ffmpeg output was decoded with the Windows locale codec and could
  raise `UnicodeDecodeError` on non-ASCII filenames.
- `setup_logging()` attached a `StreamHandler(sys.stdout)` with no encoding
  protection, so emoji log lines produce "--- Logging error ---" noise on the
  same console.

---

## 2. Files and functions changed

### New modules (`backend/autoclipper/`)

| File | Purpose | Key API |
|---|---|---|
| `duration_policy.py` | One duration contract shared by selection, planning, QA, job status and the UI (Objective 3). | `DurationPolicy.from_request(target, tolerance_ratio=0.15, shorter_allowed=True)`; `enabled`; `tolerance_seconds`; `minimum_acceptable`; `maximum_acceptable`; `verdict(actual)`; `qa_status(actual)`; `to_dict()`. Statuses: `within_tolerance` / `shortfall` / `overlong` / `unknown`; QA: `pass` / `warning` / `shortfall` / `fail`. `None` target preserves legacy behaviour as a no-op. |
| `selection.py` | Global candidate selection with duplicate/overlap/repetition penalties and evidence recording (Objective 2). | `select_distinct_candidates(pool, requested_count=, policy=, prompt=, keywords=, used_intervals=, max_overlap_ratio=0.15, max_text_similarity=0.72, allow_context_overlap=True)` → `SelectionOutcome` (`requested`/`selected`/`shortfall`/`exhausted`/`reasons`/`rejected`), plus `score_candidate`, `validate_candidates_against_prompt`, `text_similarity`. Two passes: strict, then a limited context-overlap pass that may only reuse footage for context, never to invent a distinct clip. |
| `source_profile.py` | Decide the analysis path *before* paying for transcription (Objective 4). | `measure_audio_activity()`, `classify_source(video_info, audio_activity=, transcript=, …)` → `SourceProfile` (`speech` / `audio_no_speech` / `visual_only`, `has_audio`, `has_speech`, `preserve_audio`, `uses_transcript`, `uses_visual_analysis`, `summary()`). |
| `visual_analysis.py` | Visual-only clip generation when there is no usable speech (Objective 4). | `detect_scenes()`, `build_visual_windows()`, `find_visual_candidates()` → `{candidates, scenes, mode, semantic_evidence, warning}`. Candidates always carry `visual_only: True`; the model prompt explicitly forbids inventing dialogue or captions, and when no model/key is available the result degrades to scene-boundary windows with `semantic_evidence: False` and an explicit warning instead of pretending to understand content. |

### Modified

| File | Change |
|---|---|
| `backend/autoclipper/pipeline.py` | **Objective 1:** `requested_individual_count` now always means *independent individual clips* (`clip_count`, else the legacy selected-angles count in montage-only mode) and is never reduced by `montage`. The exclusive `if montage:` branch is gone; the montage is built by the new `_build_montage()` as an **additional** output after the individual clips are planned and verified, then appended to `clips` (never counted). **Objective 3:** the render loop now records `actual_duration` from the measured file, `duration_verdict`, and `duration_shortfall`, and **downgrades a technically-passing clip to `shortfall`** when it is far below your target. **Objective 5:** new `_assess_request_compliance()` aggregates the exact clip count, per-clip measured durations, prompt relevance (`validate_candidates_against_prompt`), a duplicate/overlap re-check on the *delivered* files, and the montage's separate status into `request_compliance` (with a human-readable `failures[]`); new `_verify_montage_output()` runs post-render QA on the stitched montage as its own deliverable. **Objective 6:** the hook overlay renders to a new file, is verified non-empty, and only then replaces the clip; on any failure the prior verified render is retained and a warning is recorded. **Objective 4:** transcription is skipped entirely for a silent/no-audio source, and an empty transcript on a loud track downgrades to the visual path instead of raising. |
| `backend/app.py` | New `_job_status_for()` maps a result to `completed` **or** `completed_with_warnings`; new `_compliance_message()` explains the shortfall. Job progress now carries `request_satisfied`, `requested_individual_clips`, `delivered_individual_clips`, `duration_shortfall_clips`. New optional form field `montage_target_duration`. |
| `backend/autoclipper/hooks.py` | Replaced every `print()` with the project logger; `_decode_stderr()` and `_verify_overlay_output()` added; the overlay is written to a temporary file and only promoted after a verified non-empty result. |
| `backend/autoclipper/utils.py` | `setup_logging()` reconfigures stdout/stderr to UTF-8 with `errors="replace"` and installs `SafeStreamHandler`; `subprocess_text_kwargs()` forces `encoding="utf-8", errors="replace"` on subprocess calls. |
| `frontend/src/components/SubmitForm.jsx` | The individual clip count is no longer disabled or rewritten in montage mode; copy now states the montage is created *in addition to* the clips. Added a montage target-duration input (10–180 s), and the payload is now built by a testable pure helper. |
| `frontend/src/App.jsx` | Forwards `montageTargetDuration` → `montage_target_duration`. |
| `frontend/src/utils/directedRequest.js` | **New.** `buildDirectedClipRequest()` centralises the rule "clip count means individual clips; montage is additive" and `normalizeMontageTarget()`. |
| `backend/tests/test_montage_duration.py` | Rewritten for the decoupled contract: a montage never reduces the individual clip count, and user-selected montage moments are never dropped by text-similarity de-duplication. |

---

## 3. Tests

### Added

| File | Tests | Result |
|---|---|---|
| `backend/tests/test_request_compliance.py` | 13 — one per acceptance criterion you listed: 2 clips with montage off; 8 distinct clips when the material allows; montage adds an output without reducing the count; 60 s target is not accepted at 26.6 s; insufficient material → truthful shortfall; renamed duplicate footage is not distinct; silent source skips transcription and uses visual analysis; audio without speech does not fail the pipeline; source audio preserved with no narration/music; duration QA from measured output not planned timestamps; Unicode logging + hook-overlay failure do not crash the job; montage QA and count accounted separately. **Mocked media/model boundary.** | **13 passed** |
| `backend/tests/test_job_status_compliance.py` | 7 — the job status must reflect the request (including the exact 26.6 s vs 60 s case). Pure function tests. | **7 passed** |
| `backend/tests/test_visual_only_real_media.py` | 3 — **real media, not mocked**: ffmpeg synthesises an actual MP4 and the real pipeline renders real clips on it. | **3 passed** |
| `frontend/test/directed-request.test.js` | 3 — montage never reduces the requested clip count; montage off sends no montage fields; montage target clamped/defaulted independently. | **3 passed** |

### Mocked vs real-media coverage — read this before trusting the silent-video path

| What was tested | How | Confidence |
|---|---|---|
| Clip count decoupling, duration shortfall, montage accounting, job status | Mocked media/model boundary (`test_request_compliance.py`, `test_job_status_compliance.py`) | **Logic verified.** These drive the real `pipeline.run`, selection, planner and QA code; only ffmpeg/Groq/Gemini are stubbed. |
| Silent source → skip transcription → analyse visually → render | **Real media**: ffmpeg synthesises a 40 s MP4 with (a) no audio stream and (b) a digitally silent AAC track; the real pipeline renders real clips with real ffmpeg (`test_visual_only_real_media.py`) | **Path verified end-to-end on real files.** Observed: `transcribe` called **0** times in both cases; `analysis_mode = visual_only`; `has_audio=False` vs `True`; scene detection ran on real frames; a real non-empty `.mp4` clip was produced. |
| *Semantic* quality of visual analysis ("does the model understand the visible action?") | **NOT tested** — requires a real speech-less video plus a live `GEMINI_API_KEY`, neither of which exists in this environment | **Not validated.** Do not treat as verified. |

### Full suites

| Suite | Result |
|---|---|
| `pytest -q tests` (backend, **this sandbox as configured**) | **3 failed, 214 passed** |
| `pytest -q tests` (backend, **with a working `ffprobe` on PATH**) | **217 passed, 0 failed** |
| `npm test` (frontend, `node --test`) | **12 passed, 0 failed** |
| `npm run build` (frontend, vite production) | **built in ~4.3 s**, 1491 modules, no warnings |

#### The three "environment-related" failures, stated precisely

They are **not** code defects, and this was proven rather than assumed:

- The failures are `test_quality.py::test_ffmpeg_can_burn_captions_into_a_synthetic_clip[320x180]`, `[180x320]` and `test_quality_assurance.py::test_qa_checks_caption_sidecar_and_burned_output_artifact`.
- **Cause:** in this container `shutil.which("ffprobe")` resolves to `~/.cache/bin/ffprobe`, which is a **symlink to the `ffmpeg` binary** (`md5` identical). ffmpeg rejects ffprobe's own options with exit code 8 (`Unrecognized option 'show_entries'`). The tests therefore fail while calling ffprobe correctly.
- **Proof:** dropping in a real ffprobe implementation and re-running gives **`217 passed, 0 failed`** — every one of those tests passes unchanged. The repository's own Docker/CI image installs a genuine `ffprobe`; only this sandbox lacks one.
- They fail **identically on the untouched baseline** (`3 failed, 191 passed` there), so they are neither introduced nor affected by this work.

Verified behaviours (spot-checks beyond the suite):

```
2 clips @ 60s + montage, source supports only ~30s per topic
  requested_clip_count      : 2          individual_clip_count : 2
  montage_requested/count   : True / 1   montage_file: knife_safety_montage.mp4
  qa status                 : shortfall  compliance.satisfied : False
  montage qa                : separate block, delivery_status delivered

Duration policy @60s: tolerance ±9.0s, minimum acceptable 51.0s
  60.0s -> pass       51.0s -> pass       26.6s -> shortfall   120.0s -> warning

Hook overlay raises UnicodeEncodeError -> clip_1.mp4 retained, qa_status pass,
  warning: "Hook overlay was skipped (UnicodeEncodeError); the prior clip was retained."
```

---

## 4. Remaining limitations

1. **Visual-only analysis is scene-boundary-plus-frames, not true video understanding.**
   The current Gemini integration receives sampled frames (`Part.from_bytes`) plus
   the transcript/metadata; it never received raw video. For a silent source the
   pipeline now sends up to 12 sampled frames per window and asks only for
   visible-action descriptions, explicitly forbidding invented dialogue. With no
   API key or when the model call fails, it returns scene-boundary windows with
   `semantic_evidence: False` and a warning — it does not pretend to understand
   the content. On-screen text is read only insofar as the sampled frames show it.
   **This degradation was observed on a real file**: fed a real 40 s silent MP4
   with no detected scene boundaries, the job emitted
   *"No scene boundaries were detected; candidate windows were placed by even
   sampling and carry no content analysis"* and still rendered a real clip.
2. **Transcript-only relevance is textual.** Topic relevance is keyword/content-word
   overlap plus the edit planner's checks. It cannot reason about sarcasm, visual
   gags or off-transcript context.
3. **Duration expansion depends on the source.** The planner can only reach your
   target by including *relevant contiguous* material at sentence boundaries.
   When the source has no more relevant material it reports a shortfall. That is
   deliberate: padding with unrelated footage or repeats was explicitly rejected.
4. **The semantic visual path is unvalidated.** Only the *routing* (silent → no
   transcription → visual windows → render) is proven; whether the model picks
   good moments from real footage needs a live key and a real speech-less video.
5. **Crossfade and stitched-audio behaviour of the montage** is unchanged from
   the baseline implementation beyond the accounting and QA additions.
6. **The three ffprobe-dependent tests** cannot pass in this particular sandbox;
   they pass 217/217 with a working `ffprobe` (see §3).

---

## 5. Can 8 distinct 60 s clips be guaranteed?

**No — they can be guaranteed only in count-attempt, never in delivery.**
8 individual 60 s clips can be delivered when, and only when, the source
contains at least ~8 mutually distinct, non-overlapping story windows that are
each relevant to the prompt and each ~60 s long. AutoClipper now:

- **guarantees** an honest `individual_clip_count` separate from the montage,
- **guarantees** that the montage can never reduce that count,
- **guarantees** that a shortfall is reported per clip and in the job status,
- **attempts** the full count and duration by globally selecting distinct moments
  and expanding each to complete sentence-aligned story windows.

For a ~10-minute video that is plausible (≈8 × 60 s of distinct relevant story
windows out of 600 s) but not guaranteed; it depends on how much of the runtime
is relevant and how much is repeated. A 10-minute video with 4 minutes of
relevant, non-repeating material will truthfully deliver fewer/shorter clips
rather than pad.

## 6. Safe manual test with the knife-safety video

1. Use a **throwaway output directory** (or a copied source); nothing here
   deletes your original file.
2. Start the app, upload the knife-safety video, and let the analysis finish.
3. In Step 2:
   - set **Individual clips to generate = 2** (the selector is now enabled in montage mode),
   - set **Target duration = 60**,
   - optionally tick **Build a multi-shot montage** and set **Montage target = 60 s**,
   - optionally tick **Add hook** (this is the path that raised the Windows error).
4. Start the job and watch the progress messages. Expected outcomes:
   - success case: `individual_clip_count = 2`, each clip ≈51–69 s, `qa_status = pass`;
   - shortfall case: each clip is ~26–30 s, each carries a duration warning, the
     QA status is `shortfall`, and the **job status is `completed_with_warnings`**
     with a message naming the delivered vs requested length;
   - a montage, if requested, appears as a **separate file** with its own
     `montage_report` and never changes `individual_clip_count`.
5. Check the returned `processing_warnings`. If the hook overlay could not be
   rendered, you will see `Hook overlay was skipped (…)` and the clip will be the
   verified pre-overlay render — the video is never left half-written.
6. On Windows, confirm no `UnicodeEncodeError` appears in the console output.

## 7. Recommendation

**Ready for another manual test — but not for merge or deploy yet.**
The count/duration/accounting defects you hit are fixed, guarded by 26 new
automated tests (23 mocked + 3 real-media), and the job status no longer
misreports a shortfall as success. Keep two caveats in view while testing:

- **Needs more work** before "8 guaranteed 60 s clips" is a promise we can make —
  that is a property of the source, not of the code. The honest shortfall is the
  correct behaviour.
- **The semantic visual path is unvalidated.** Its routing is proven on real
  media; its clip-picking quality is not. Test it separately with a real
  speech-less clip and a live key.

**Not blocked.** No dependency on PR #8 (already merged); all work is on the
`arena/c6ac8e24-autoclipper` feature branch, delivered as an open, unmerged PR.

### Acceptance test to run first

Request **two 60-second clips**. The pass condition is two *independently
planned* clips each targeting 60 s — never two 30 s pieces produced because
montage mode divided the duration budget. If the source cannot support two
distinct, relevant 60 s clips, the job must say so: `qa_status = "shortfall"`,
`request_compliance.satisfied = false`, and a job status of
`completed_with_warnings` naming the delivered vs requested length.
