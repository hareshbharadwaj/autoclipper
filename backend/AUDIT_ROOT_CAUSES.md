# AutoClipper audit — root causes of poor clips (knife-safety reproduction)

Branch: `arena/8187ed97-autoclipper`, audited at commit `6824f80` (main).

## Reproduction case (analyzed from code; original media NOT available here)

- Source: `Knife Safety and Skills - 10 min video 480P.mp4` (~604 s, 854×480) — **not present in
  this environment** (media files are git-ignored and no cached copy exists).
- Selected angle: `3 Knife Rules You NEED to Know!`, hints ≈ 91–117 s (26 s span).
- Request: 60 s target, 1 clip, 16:9, custom direction about knife safety rules.
- Previous run `903eec2ee4e3` produced **84.2–144.0 s (59.8 s)**; candidate review failed with a
  client error and the pipeline retained the widened timestamps.

The exact code path that produced that expansion (verified by reading `pipeline.py`/`analyze.py`):

1. `pipeline.run` resolves `target_duration=60` → hard window `57–63 s`
   (`analyze._resolve_clip_bounds`).
2. The user-selected angle is passed through `analyze._validate_clips(min_clip=57, max_clip=63)`,
   which accepts only durations in `[57-2, 63+3]` seconds. The 26 s selection **fails the window
   and is silently dropped**.
3. Because `selected_candidates` is now empty and `clip_count` was requested, the pipeline calls
   `find_viral_clips` again, with the angle title demoted to "creative direction" and a hard
   57–63 s duration instruction. Gemini must now invent boundaries that fit 60 s and chose
   84.2–144.0 s — padding ~7 s of lead-in and ~27 s of trailing material whose relevance to the
   requested rules was never checked.
4. The independent review (`review_clip_candidates`) raised a client error; the pipeline caught it,
   kept status `fallback_unreviewed`, and retained the widened timestamps with a warning.
5. Post-render QA checked only technical properties (streams/duration/aspect/audio/subtitles/frame
   decode). Nothing verified that 84.2–144.0 s actually contained the requested safety rules, that
   the opening was not mid-thought, or that the ending resolved the topic.

## Root-cause list

| # | Root cause | Where |
|---|-----------|-------|
| RC1 | User-selected angle boundaries are discarded when they don't fit the target-duration window; the hint span is lost before generation | `pipeline.run` (selected branch) + `analyze._validate_clips` |
| RC2 | Target duration enforced as a hard window; shorter-but-accurate clips are filtered out and generation is forced to pad with adjacent (possibly unrelated) material | `analyze._resolve_clip_bounds`, `analyze._filter_clips_to_bounds`, `find_viral_clips` prompt |
| RC3 | No factual-grounding QA anywhere: evidence quotes are stored unverified by `find_viral_clips`; titles/hooks are never checked against the clip's transcript span; no negation/qualifier/boundary checks | entire pipeline |
| RC4 | Review failure collapses every failure mode (timeout, malformed, client error) into one broad `except Exception` → single `fallback_unreviewed` state; final clips carry no per-clip review/verification status | `pipeline.run`, `analyze.review_clip_candidates` |
| RC5 | "Review unavailable" candidates flow downstream indistinguishably from reviewed ones; nothing enforces that unreviewed candidates must still pass mandatory factual checks | `pipeline.run` |
| RC6 | Transcript-only paths can be reported as visual verification: review `visual_evidence` is free model text, unchecked against whether frames were actually sampled for that candidate | `analyze.review_clip_candidates`, `pipeline.run` |
| RC7 | No deterministic boundary semantics: nothing checks that cuts align to sentence boundaries or preserve negations/conditions/warnings around cuts | pipeline (absent) |
| RC8 | Individual clips never report requested-vs-actual duration with a reason (only montage did) | `pipeline.run` |
| RC9 | Generated hook text is burned without any grounding check (could fabricate a quotation) | `pipeline.run` hook stage |
| RC10 | No unified source-of-truth record (SourceAsset/ClipPlan-style provenance): final metadata lacks the exact transcript span used per clip | `pipeline.run` result |

## What already works (must be preserved)

- Transcript-grounded review that derives boundary moves only from exact segment quotes
  (`analyze.review_clip_candidates`).
- Honest technical QA with `pass/warning/fail` statuses and explicit fallback restoration
  (`quality_assurance.verify_rendered_file`, pipeline fallback chain).
- Montage fitting on complete transcript segments with duration-deviation warnings.
- Subtitle pipeline builds SRT from actual word timestamps (verbatim), resolution-aware layout.
- Cached-transcript reuse between analysis and processing.
