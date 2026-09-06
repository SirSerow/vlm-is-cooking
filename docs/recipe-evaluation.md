# Training-video recipe and next-step evaluation

Evaluated on 2026-09-06 using the user's training footage. **The app is not yet
reliable at deciding when to suggest the next cooking step.** It matched 25/35
expected recommendations (71.4%), but made six premature suggestions and did not
get any of the seven complete transition cases right.

## Recipe reconstructed from the footage

The dish appears to be **creamy chicken with mushrooms and onion**. The source is
`S:/Projects/cooking-detection-project/datasets/video/original_uncut.mp4`, about
36 minutes 18 seconds long. Visual review used three frames per episode from the
13 chronological, rotated clips derived from that source. Episode metadata
provided timestamps and ingredient names; the images verified the visible
actions. This was an agent-authored reconstruction, not an automatic recipe
compiler built into the application.

| Source-video interval | Visible workflow |
| --- | --- |
| 00:00–00:40 | An empty pan/work surface is prepared. |
| 00:40–01:49 | Mushrooms are sliced and collected on a plate. |
| 01:49–02:10 | An interlude labeled plate washing; washing itself is outside the inspected crop. |
| 02:10–02:50 | Onion is chopped and transferred to a small bowl. |
| 02:50–05:30 | Chicken is cut into pieces. |
| 05:40–07:30 | Oil is added to the pan on the hob. |
| 07:30–14:21 | Mushrooms are added, stirred and cooked. |
| 14:21–19:36 | Chopped onion is added and cooked with the mushrooms. |
| 19:36–26:08 | Chicken pieces are added, turned and cooked; their exterior becomes paler. |
| 26:08–29:54 | A pale cream-like liquid is added and stirred into the contents. |
| 29:54–32:08 | The pan is covered. |
| 32:08–35:48 | The lid is removed and the contents are stirred. |
| 35:48–36:18 | The pan is covered again; the recording ends. |

These are episode spans, **not prescribed cooking times**. Ingredient weights,
exact dairy product, seasoning amounts, heat settings and final doneness are not
established. No serving or thermometer check is shown. The executable recipe
adds a clearly identified final manual doneness check; it is not claimed to have
occurred on video. The 05:30–05:40 gap is outside the supplied episode clips.

The [recipe JSON](../recipes/creamy-chicken-mushrooms.json) separates visible
ingredient transfers and lid changes from preparation, heating and cooking
steps that need confirmation. All steps require explicit confirmation to advance.
Visible presence alone cannot establish that an entire portion was transferred
or that stirring is finished.

## What was implemented and tested

The previous app only returned observations. This change adds `Recipe`, `Step`
and `RecipeSession`, a small interactive cooking CLI, and a reproducible offline
video benchmark. The observer receives the image and current-step queries; it
never receives the clip filename, timestamp, expected label, next instruction
or whole recipe. Recommendations come from recipe order and kitchen state.

For a visual step, the app suggests the next instruction when two of the last
three distinct observations match its expected state with confidence at least
0.7, within 30 seconds, and the latest observation also agrees. Unknown,
low-confidence, stale and contradictory observations do not produce readiness.
Confirmation clears the previous step's votes. The model never advances a step.

The benchmark covers seven cases: adding mushrooms, onion, chicken and cream;
covering, uncovering and covering the pan again. Each case contains two negative
pre-action frames and three positive post-action frames, sampled three seconds
apart on the positive side. All 35 images were visually inspected and labeled
before model inference. The [frozen sampling manifest](../config/recipe_evaluation.v1.json)
records clip names, timestamps and the fixed crop `(500, 330, 1120, 1080)`.

The current step is supplied at the start of each case, simulating the user
confirming preceding actions. Cases are independent; there is no hidden
resynchronization within a case. A correct decision stays on the current step
until two fresh positive frames have arrived, then suggests the next instruction.
The scorer computes that expectation from the visual labels and sampling times,
not from the model's answers. Invalid responses count as failures.

**This is a supervised transition benchmark, not a full autonomous video replay.**
It does not test recognizing the recipe position from scratch, completeness of
preparation, heat readiness, cooked-through chicken, camera scheduling, or a user
following the resulting advice. Only the VLM is active; the locally trained YOLO
detector is not yet integrated into the application. The footage overlaps the
YOLO training session, and earlier VLM smoke tests also used this session.

## Results

Model: `qwen3-vl:2b-instruct`, Q4_K_M, through Ollama 0.33.3. Model digest:
`ea422f1e73652a95479954d8572d3c8c6022f628ce2d38a1a04aae1b7f2d5300`.
Temperature 0, seed 42, context 4096, output limit 512. No model retraining,
prompt tuning, threshold tuning or failed-frame replacement was performed during
this benchmark.

| Measure | Result |
| --- | --- |
| Structurally valid model responses | 35/35 (100%) |
| Correct visual precondition/postcondition classification | 20/35 (57.1%) |
| Correct next-step decision | 25/35 (71.4%) |
| Baseline: always keep the current instruction | 21/35 (60.0%) |
| Baseline visual classifier: always say the action happened | 21/35 (60.0%) |
| Positive frames recognized | 16/21 (76.2%) |
| Negative frames incorrectly called positive | 10/14 (71.4%) |
| Premature next-step decisions | 6 |
| Fully correct five-frame cases | 0/7 |
| Mean inference request time | 1.283 seconds |

The recommendation score is higher than visual accuracy because waiting for a
second vote temporarily masks some wrong answers. Six premature recommendations
include five on negative frames and one on the first positive frame, where a
preceding false positive supplied an extra vote. Correlated mistakes defeat
two-frame voting. Reported confidence on false positives was 0.95–1.0, so high
model confidence was not a reliable correctness signal.

| Visual step | Correct next-step decisions | Main failure |
| --- | --- | --- |
| Add mushrooms | 4/5 | Reports mushrooms in the empty, oiled pan. |
| Add onion | 4/5 | Reports onion before it is added. |
| Add chicken | 4/5 | Reports chicken in the mushroom/onion mixture before addition. |
| Add cream | 4/5 | Reports cream before the liquid is added. |
| Cover pan | 3/5 | Misses the lid on two of three covered frames. |
| Uncover pan | 3/5 | Reports the lid off while it is still covering the pan. |
| Cover again | 3/5 | Reports the lid off in all three covered frames. |

## Failure examples

At source time **07:25**, the pan contains oil and no mushrooms. The observer
returns `mushroom.location = pan`, confidence 0.95. Because the preceding frame
has the same false answer, the app incorrectly recommends cooking mushrooms.

![Two empty-pan frames followed by three mushroom frames](../outputs/recipe-evaluation/benchmark/review-add_mushrooms.jpg)

At **30:07**, a lid visibly covers the pan. The model returns
`lid.position = off_pan`, confidence 1.0. The app continues asking for the lid to
be placed instead of suggesting the covered cooking step.

![Uncovered pan followed by three covered-pan frames](../outputs/recipe-evaluation/benchmark/review-cover_pan.jpg)

Two additional, unscored description requests explored these errors. With an
open description prompt, the model correctly described the empty pan as
containing oil, but still failed to recognize the covering lid. This suggests
both a structured-query grounding problem and a visual recognition problem;
it does not establish that a different prompt fixes the benchmark.

## Interpretation and next development step

The recipe ordering is usable as a **manually confirmed checklist**. The current
model's visual evidence is too unreliable to control cooking progression or
consistently time next-step suggestions. Schema-valid JSON is not sufficient.
Repeated confidence scores cannot compensate for systematic recognition errors.

The next useful experiment is explicit ingredient-presence verification before
location queries, alongside the trained detector for lid/pan evidence. Any
changes should keep this run as a baseline and be evaluated on additional
negative frames and a separate cooking session. A describe-then-extract observer
is also worth testing, but was not implemented or scored here.

## Artifacts and reproduction

- [Executable inferred recipe](../recipes/creamy-chicken-mushrooms.json)
- [Sampling manifest and labels](../config/recipe_evaluation.v1.json)
- [Recorded per-frame results and model identity](recipe-evaluation-results.json)
- [Full raw observations, timings and recommendations](../outputs/recipe-evaluation/benchmark/results.json)
- [Evaluation runner](../scripts/evaluate_recipe.py)
- [Interactive usage and rerun commands](../README.md)

Source images, contact sheets and raw outputs remain in gitignored `outputs/`;
the recipe, manifest, compact results and this report are included in the project.
The application and recipe tests pass: 11 tests covering response validation,
state freshness, distinct-frame voting, contradictions and manual progression.
