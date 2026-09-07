# VLM size and hybrid architecture evaluation

Date: 2026-09-07. Status: plan; new experiments have not been run.

The decision is whether a larger VLM improves cooking recognition enough to justify its cost, and whether YOLO plus tracking helps deliver that accuracy with fewer missed events or less computation. A slower model alone does not justify a hybrid architecture.

## 1. Models and controlled comparisons

| Size | Exact initial candidate | Question |
| --- | --- | --- |
| 2B | `qwen3-vl:2b-instruct-q4_K_M` | Reproduce the current small-model baseline. |
| 4B | `qwen3-vl:4b-instruct-q4_K_M` | Does a moderate size increase fix the observed errors at usable latency? |
| 8B | `qwen3-vl:8b-instruct-q4_K_M` | Is there a substantial accuracy gain worth slower execution or different hardware? |

These variants are listed in the [Ollama model registry](https://ollama.com/library/qwen3-vl/tags), checked 2026-09-07. Use explicit instruct/quantization tags and record downloaded digests. Avoid default tags that may select a thinking variant. Start with this same-family comparison; another family is a separate experiment, not a model-size result.

The documented machine is an RTX 3060 Laptop GPU with 6 GiB VRAM. Registry download sizes are about 1.9/3.3/6.1 GB respectively; these are not runtime memory requirements. Measure actual memory and CPU offload. In particular, do not assume 8B fits alongside YOLO. A failed fit is a deployment result, not an accuracy score. Larger-GPU runs may establish recognition quality but must appear in a separate hardware group for latency comparisons. No remote provisioning is required by this plan.

Keep image pixels, query wording, schema, temperature 0, seed 42, context 4096, and output limit 512 fixed for the initial single-image comparison. Freeze code and prompt hashes. Record any required configuration change as a separate run. Use the same metrics and acceptance targets for every size; larger models do not get easier accuracy criteria.

## 2. Run in three stages

### A. Reproduce and screen model quality

Run all three sizes on the existing 35 frozen images using `scripts/evaluate_recipe.py --model ...`, with separate output directories containing identical sample manifests and image hashes. Its `--reuse-samples` option expects samples inside that run's output directory. Preserve the historical run.

This is a regression screen, not evidence of unseen-session performance or brief-event recognition. Report raw allowed-value correctness separately from the existing confidence-thresholded visual-match metric: the current runner combines correctness with a confidence threshold of 0.7.

Next use a reviewed development set with empty/oiled pans, look-alike mixtures, lids beside versus on pans, partial transfers, and occlusions. Allow prompt development only here, equally budgeted across sizes. Freeze the final prompt before the test set.

### B. Compare size on held-out evidence

Target at least three new cooking sessions, 60 observable transitions total, and 60 matched no-event intervals, with at least 10 events each for lid placement, lid removal, solid transfer, and liquid transfer. Report actual counts and gaps if collection falls short. Include brief events under three seconds and long unchanged periods. No session used for detector training, prompt tuning, or threshold selection belongs in the test set.

Review videos before model inference. Label event type, start/end, affected container, visible ingredient when identifiable, and the earliest time the visual postcondition is established. Label partial transfers and unobservable evidence explicitly. Ingredient presence does not establish that the entire portion was transferred. Resolve ambiguous labels before freezing the manifest; report excluded and unobservable counts.

For every size, score identical single images and identical three-image sequences in separate tables. This separates model-size gains from gains due to temporal context. Sequence input requires extending the current single-image observer. Do not substitute a montage silently; if needed, freeze and name that input format separately.

### C. Test the hybrid on continuous replay

Run each locally feasible size in these modes:

| Mode | Evidence selection | Purpose |
| --- | --- | --- |
| A: periodic image | Newest image at three-second scheduling slots | Simple baseline |
| B: periodic sequence | Three ordered frames at now minus 3, 1.5, and 0 seconds | Is buffering alone sufficient? |
| C: hybrid sequence | YOLO/tracker candidate events select three ordered frames; periodic sequence fallback every three seconds | Does event selection add value? |

Use a 15-second rolling buffer. For an event candidate, select the last frame before its detected start, a middle frame, and the latest available frame; retain original timestamps. Freeze candidate rules on development footage (initially lid/pan overlap changes and container approach/departure). These are interaction candidates, not confirmed pours. Keep YOLO weights, tracker settings, input resolution, and 10 FPS target fixed across sizes. Test 30 FPS only as a separately named follow-up if short events are missed.

Replay at wall-clock speed, with one VLM request in flight. Retain at most one pending event sequence and one replaceable periodic request; event requests have priority. Merge overlapping candidates for the same interaction, and log candidates dropped when capacity is exhausted. Drop missed periodic slots instead of queuing them. A slow response becomes available only when inference completes: never apply it retroactively. Prevent old evidence from overwriting newer state.

Keep recipe progression supervised and the current step supplied identically across modes. Use the existing two-distinct-observation rule for recommendations, with freshness checked at response time. Overlapping sequences reusing the same postcondition frame cannot count as independent confirming observations. Also score semantic event recognition separately from recipe readiness.

Perform an equal-budget comparison of B and C: on development footage choose scheduling intervals giving total measured GPU busy time per video minute within 10%, then freeze and replay held-out sessions. Count detector work in C's budget. Report achieved resource use; if budgets cannot be matched, label the comparison unmatched. Calls per minute alone are not compute because sequences and models differ in cost.

## 3. Metrics: what each number means

Always show numerator/denominator alongside percentages. Unknown and invalid responses must not disappear from the denominator.

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Visual accuracy | Correct allowed-value answers / all scorable labeled queries | Does it see the right state? Unknown on a determinate label and errors count wrong. |
| Positive recognition | Correct positive answers / all positive queries | Does it notice a state that really exists? |
| False-positive rate | Positive answers / all negative queries | Does it invent ingredients or completed states? |
| Balanced accuracy | Mean of positive recognition and correct-negative fraction | Prevents a majority class from hiding errors; unknown/errors are correct for neither class. |
| Abstention / invalid rate | Unknown answers / all queries; failed responses / all requests | Distinguishes uncertainty from broken output. Score labeled unobservable cases separately. |
| Event recall within 10 s | Correct events reported by 10 s after labeled event end / all observable events | How many actions were recognized promptly? |
| Event precision | Correct one-to-one matched event reports / all event reports | How often an announced event actually happened. |
| False suggestions per hour | Premature or unsupported readiness episodes / scored video hours | How often the assistant incorrectly says to continue. |
| Successful transitions | Cases with no premature readiness and correct readiness within 10 s of visual postcondition / all cases | Primary recipe-level usefulness measure. |
| Recognition delay, median/p95 | Response-available time minus event-end time, for matched events | Waiting time after an action; display missed events beside this metric. |
| Warm request latency, median/p95 | Dispatch to validated response, including preprocessing/transport | Real request cost; p95 means 95% completed within this time. |
| Three-second deadline success | Requests completed within 3 s / all requests | Whether the requested cadence is realistic. Also report skipped slots. |
| Resource use | Peak total VRAM, peak process RAM, offload fraction, calls/minute, GPU busy seconds/video minute | Whether the configuration fits and what it costs. |
| Trigger coverage (hybrid) | Events with a selected sequence overlapping the labeled action / all events | Separates YOLO selection failures from VLM interpretation failures. |

Event matching requires matching type and target container, supporting frames overlapping the annotated action, and a report available between event end and end + 10 seconds. Use one-to-one matching. Repeated identical reports within one continuous event episode collapse to one report; additional unmatched reports count false. Report late correct events separately and include them as deadline misses. Preserve raw reports for auditing.

Count a readiness episode on a not-ready to ready change, including a wrong ready state at case entry; repeated ready outputs do not inflate the count. Ground-truth postcondition time, not model votes, determines whether readiness is premature. Also report the fraction of cases with any premature suggestion.

Measure three warmed replays per feasible configuration in rotated order and one cold-start request separately. Timing repetitions do not create additional independent accuracy examples. Record decode, queue wait, inference, and total event delay separately. Run a 30-minute concurrent stability check on finalists. Use changed frames; disclose cache behavior. Timeouts, OOMs, truncations, and skipped work remain in the report.

Break down results by event type, brief versus persistent event, session, and visibility. For comparisons use paired resampling by session for 95% intervals, marking intervals exploratory with only three sessions. Never treat neighboring video frames as independent statistical samples.

## 4. Decision rules fixed before testing

These are proposed engineering targets, not already demonstrated performance or safety certification:

* A candidate is promising for a supervised demo at event recall >=90%, event precision >=95%, <=1 false readiness episode/hour, p95 event delay <=10 seconds, and invalid responses <=1%. All targets must hold; show session-level failures. Small datasets may leave the decision inconclusive.
* Prefer a larger model when paired balanced accuracy improves by at least 5 percentage points without increasing false positives. If uncertainty includes no improvement, report a promising trend rather than a proven size benefit.
* Prefer hybrid C over buffered B at the same model size if event recall improves by >=5 percentage points under matched compute, with no increase in false suggestions and no worse p95 delay; alternatively, require >=25% less measured total GPU busy time at recall within 2 percentage points and unchanged error/delay targets.
* A model that improves accuracy but cannot meet local memory or delay targets is an accuracy reference, not the local deployment winner. If B matches C, choose the simpler buffered VLM architecture.

## 5. Understandable output

Produce `report.md`, `model-summary.csv`, `pipeline-summary.csv`, `events.csv`, and raw `requests.jsonl` in one versioned run directory. Missing values are `N/A` with reasons, never zeros. The first page answers: Does bigger help? Does YOLO help beyond buffering? Which configuration fits this laptop?

Model-size summary (all rows must use the same dataset, input format, and hardware group):

| Size | Right answers | False positives | Unknown | Typical / p95 request | Within 3 s | Peak VRAM / offload | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2B | Not run | Not run | Not run | Not run | Not run | Not run | Pending |
| 4B | Not run | Not run | Not run | Not run | Not run | Not run | Pending |
| 8B | Not run | Not run | Not run | Not run | Not run | Not run | Pending |

Pipeline summary (one row per size/mode/hardware):

| Size + mode | Actions caught within 10 s | False suggestions/hour | Typical / p95 delay | GPU busy s/video min | Dropped candidates | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Each feasible configuration | Correct/total (%) | Count/hours | Seconds + miss count | Measured | Count | Pass / fail / inconclusive |

Include an accuracy-versus-delay scatter plot, colored by mode and labeled by model size, plus a timeline for each failed transition: actual event, selected frames, request start/end, model answer, and recipe suggestion. Show before/during/after thumbnails and plain-language failure categories: missed selection, wrong interpretation, stale answer, or incorrect recipe rule.

Historical context only: the existing 2B screen recorded 20/35 confidence-thresholded visual matches, 10/14 negative frames called positive, 6 premature frame-level suggestions, 0/7 fully correct cases, and 1.283 s mean request time. Its six suggestions are not the new deduplicated episode metric. No p95, continuous event recall, or held-out accuracy is established by that run. See [the original report](recipe-evaluation.md).

## 6. Work required to execute

1. Reproduce the frozen screen for 2B, 4B, and 8B; record fit/offload and model identities.
2. Collect and review new sessions; freeze temporal labels and development/test split.
3. Extend the observer for timestamped multi-image evidence and the runner for raw semantic scoring, request telemetry, and summary export.
4. Implement buffered replay, tracking candidates, bounded scheduling, and response-time state updates.
5. Run the controlled size study, then replay and equal-budget comparisons; produce tables, timelines, and verdicts.

The existing runner covers the single-image regression screen only. The multi-image observer, temporal labels, integrated tracker, causal replay, telemetry, and new report generator remain to be implemented. This plan does not claim those capabilities already exist.

## 7. Architecture changes required for the tests

The full evaluation requires extensions, but the existing separation between observation, timestamped kitchen state, and recipe logic can stay. The main missing capability is temporal evidence handling: the app currently processes one still image per request. Running YOLO beside the VLM does not yet influence its inputs or recipe decisions.

| Evaluation | Required change | Existing component to preserve or extend |
| --- | --- | --- |
| 2B versus 4B versus 8B on frozen images | No architecture change; use the existing `--model` argument. Add latency distributions, memory/offload measurements, and comparable reports. | `scripts/evaluate_recipe.py` and `OllamaObserver` |
| VLM on buffered sequences | Add a rolling video buffer and an observation input containing multiple ordered, timestamped frames. Preserve the single-image path for baseline runs. | `Frame`, `Observer`, and `OllamaObserver` |
| YOLO plus tracking plus VLM | Connect detections to a tracker, derive candidate interactions, and select evidence from the buffer. Candidates remain hypotheses until evaluated. | A new evidence-selection layer before the observer |
| Realistic speed and delay comparison | Add one in-flight VLM request, bounded pending work, skipped-slot accounting, and response-time application of results. | Replay scheduler around `CookingAssistant` |
| Fair recipe evaluation | Retain source-frame identities for each sequence, prevent reused evidence from supplying independent votes, and check freshness at response time. | `Observation`, `KitchenState`, and `RecipeSession` |

Keep event time, source-frame capture times, and response-available time distinct. A sequence cannot be represented faithfully by a single timestamp alone. Preserve evidence provenance through observation and state updates so the scorer can identify stale answers and duplicate evidence.

Implement and evaluate in this order:

1. **Model-size screen:** determine whether larger VLMs fix current recognition errors before building the hybrid integration.
2. **Buffered sequence baseline:** determine how much improvement comes from temporal context alone, using identical sequences across model sizes.
3. **YOLO-driven selection:** measure the additional benefit of tracking and candidate-event selection against the buffered baseline, including detector compute.

This order allows each added component to justify its contribution. The first test can start with the current application; the complete hybrid test depends on the extensions above.
