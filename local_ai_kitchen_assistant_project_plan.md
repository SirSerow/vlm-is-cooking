# Local AI Kitchen Assistant — Project Plan

Implementation review: see [docs/plan-review.md](docs/plan-review.md) for required
corrections and validation gates. Remote annotation setup is documented in
[environments/sam3/README.md](environments/sam3/README.md).

## 1. Goal

Build a local overhead-camera kitchen assistant that can:

- Load a recipe from a URL or pasted text.
- Compile the recipe into structured, machine-readable steps.
- Observe the kitchen periodically.
- Use YOLO for a fixed set of kitchen tools and containers.
- Use a local VLM as a structured observer that answers recipe-compiled queries.
- Merge both into `KitchenState`; never let a model advance the recipe directly.
- Advance the recipe when `KitchenState` matches a step's `expected_after`.
- Suggest cleanup actions when tools or ingredients are no longer needed.

The first version should prioritize simplicity over real-time processing.

---

## 2. High-Level Architecture

```text
Recipe URL / Text
        │
        ▼
Recipe Extractor
        │
        ▼
Local LLM Recipe Compiler
        │
        ▼
Executable Recipe JSON
        │
        ├─ session entities
        └─ current-step queries
                │
                ▼
        Perception Manager
           ┌────┴────┐
           │         │
         YOLO      Local VLM
           ▲         ▲
           └────┬────┘
                │
        Overhead Camera
                │
                ▼
          Kitchen State
                │
                ▼
      Recipe State Machine
      (compare to expected_after)
```

---

## 3. Camera Acquisition

The camera can capture continuously at its normal frame rate, for example 30 FPS, but inference should not run on every frame.

Only the latest frame needs to be stored.

```text
Camera thread
    │
    ├─ capture frame
    ├─ replace latest_frame
    └─ repeat
```

The inference loop requests the newest frame when needed.

This avoids unnecessary buffering and latency.

---

## 4. Inference Modes

### Manual Mode

The first implementation should allow the user to manually trigger analysis of the latest camera frame.

Use this to test:

- Camera placement
- Lighting
- Query generation from a step
- Closed-vocabulary VLM answers
- Ingredient visibility
- YOLO vs VLM labor split

### Periodic Mode

After manual mode works, automatically analyze the newest frame at a fixed interval.

Initial target:

```text
VLM interval: 3 seconds
YOLO interval: 0.5 seconds
```

Both models can initially process the same sampled frame.

Later, YOLO can run more frequently if needed.

Example future configuration:

```text
Camera: 30 FPS
YOLO: 1–2 FPS
VLM: every 3–5 seconds
```

YOLO model is used to judge if the scene has changed and it is time to run more computationally intensive VLM.


### Input modes
The inference should be possible with a camera stream input and the regular video stream input when camera is not available.

---

## 5. Local VLM

The VLM is a structured observer, not a recipe judge.

It never decides that a step is complete. It only answers compiled queries about the current frame. Those answers update `KitchenState`. The recipe state machine compares `KitchenState` to `expected_after`.

```text
observe(frame, yolo, queries) -> answers
```

### What the VLM must not do

- Invent JSON keys per step (`pan_visible`, `onion_location`, ...).
- Answer questions YOLO already answered with high confidence.
- Return `step_completed`.
- See the whole recipe. Current-step entities are enough.

### Query generation

Queries are compiled from the current step, not written by hand.

From:

```json
"expected_after": {
  "onion": {
    "location": "pan",
    "state": "chopped"
  }
}
```

Generate:

```json
{
  "queries": [
    {
      "id": "onion.location",
      "entity": "onion",
      "ask": "location",
      "allowed": ["prep_area", "cutting_board", "pan", "bowl", "not_visible", "unknown"]
    },
    {
      "id": "onion.state",
      "entity": "onion",
      "ask": "state",
      "allowed": ["whole", "chopped", "unknown"]
    }
  ]
}
```

Ask about:

- current step entities
- optionally next-step entities
- leftover tools during cleanup

Do not ask about the whole recipe. Typical query count: 2–6.

Skip any query YOLO already answered. If YOLO says `pan` is present at the stove, do not ask `pan.visible`.

### Stable output schema

Every recipe uses the same response shape:

```json
{
  "answers": [
    {
      "id": "onion.location",
      "value": "pan",
      "confidence": 0.86
    },
    {
      "id": "onion.state",
      "value": "chopped",
      "confidence": 0.74
    }
  ],
  "unknown": ["salt.quantity"],
  "anomalies": ["oil pooling on counter"]
}
```

`unknown` is load-bearing. Overhead cameras cannot see salt quantity, a closed lid, or 1/2 teaspoon. Unobservable fields fall back to `user_confirmation` instead of hallucinating completion.

Values must come from the query's `allowed` list. If the model cannot map what it sees onto that list, it returns `unknown`.

### Prompt shape

Local VLMs on a 3060 are bad at long prose and invented keys. Keep prompts short and multiple-choice.

```text
Overhead kitchen frame.
YOLO: pan[420,180,710,480], knife[120,350,280,390]

Answer each query. Use only allowed values.
If unsure, unknown.

1. onion.location  allowed=[prep_area, cutting_board, pan, not_visible, unknown]
2. onion.state     allowed=[whole, chopped, unknown]

JSON only:
{"answers":[{"id":"...","value":"...","confidence":0.0}]}
```

If a single structured call is still flaky, use a two-pass that still ends in the same schema:

1. VLM: short scene description plus YOLO boxes
2. Text LLM: extract answers into `answers[]`

The application consumes `answers[]`, never free text.

The local VLM can initially be run through Ollama on the RTX 3060.

---

## 6. Closed Vocabularies

There are three vocabularies with different lifetimes. “What YOLO can detect” and “what this recipe cares about” are not the same list.

```text
YOLO classes          = kitchen furniture   (fixed model)
Recipe entity list    = this cook's nouns   (from compiler)
Attribute vocab       = legal answers       (fixed schema)
```

### Kitchen objects — closed forever (YOLO)

YOLO detects kitchen infrastructure, not ingredients. Its class dictionary is fixed for a trained model version and must not change order between annotation, training, evaluation, TensorRT export, and runtime.

V1 annotation dictionary (15 classes):

| Id | Class | Annotation boundary |
|---:|---|---|
| 0 | `pan` | Frying and saute pans; include the handle. |
| 1 | `pot` | Sauce and stock pots; exclude woks. |
| 2 | `wok` | Woks and similarly deep, rounded pans. |
| 3 | `lid` | Loose pan or pot lids, including when resting on a vessel. |
| 4 | `knife` | Kitchen knives; exclude scissors and peelers. |
| 5 | `cutting_board` | Boards used as a preparation surface. |
| 6 | `bowl` | Mixing, prep, and serving bowls. |
| 7 | `plate` | Flat plates and shallow serving dishes. |
| 8 | `spatula` | Turners and silicone cooking spatulas. |
| 9 | `spoon` | Cooking and table spoons; exclude ladles in V1. |
| 10 | `tongs` | Kitchen tongs. |
| 11 | `cup` | Cups and measuring cups. |
| 12 | `bottle` | Oil, sauce, and squeeze bottles; exclude jars. |
| 13 | `jar` | Glass or plastic jars with a wide mouth or screw lid. |
| 14 | `sponge` | Sponges and scrub pads. |

The dictionary intentionally excludes ingredients, food packages, hands, utensils not listed above, and vague classes such as `food` or `clutter`. They are either recipe-specific VLM entities or later SAM 3 prompts.

Treat this table as an annotation contract. Ambiguous cases are better marked `ignore` than forced into the wrong class. Add a new class only through a new dictionary and model version, followed by retraining and full re-evaluation.

Rule:

> If it is a container or tool that lives in the kitchen, it belongs to YOLO.
> If it is an ingredient that arrives with the recipe, it does not.

Do not add `onion`, `garlic`, `tomato`. Do not retrain YOLO per recipe.

Rare tools (`mandoline`, `microplane`) stay out of YOLO. Treat them as a VLM entity for that session, or verify the step with `user_confirmation`.

Example YOLO output:

```json
{
  "pan": {
    "present": true,
    "bbox": [420, 180, 710, 480]
  },
  "knife": {
    "present": true,
    "bbox": [120, 350, 280, 390]
  }
}
```

Coarse location can come from geometry (bbox vs stove / prep / sink zones) without the VLM.

### Recipe entities — closed for this cook (VLM)

This set changes with the recipe, but it is compiled once, not invented at inference time.

```text
session_entities = objects_required ∪ ingredients in expected_before/after
```

Example for “add chopped onions to the pan”:

```json
{
  "session_entities": {
    "objects": ["pan", "knife", "cutting_board"],
    "ingredients": ["onion"]
  }
}
```

The VLM may only talk about this list, plus `unknown`. At each tick, shrink it to current-step entities (2–6 things), not the whole recipe.

Later, an open-vocab detector (YOLO-World, Grounding DINO, SAM 3) can be prompted with the same session entity list. That is still closed per cook. It is not a reason to retrain YOLO.

### Attributes — closed globally

These almost never change with the recipe.

Locations:

```text
stove, prep_area, sink, cutting_board, pan, pot, bowl, plate, not_visible, unknown
```

Ingredient states:

```text
whole, chopped, sliced, minced, mixed, raw, cooking, browned, melted, unknown
```

Presence:

```text
true, false, unknown
```

The recipe says which entity to inspect. The VLM only chooses from these values.

### Name canonicalization

Recipes will say “yellow onion”, “onions”, “diced onion”. KitchenState always uses a canonical id.

```json
{
  "id": "onion",
  "aliases": ["onion", "onions", "chopped onion", "diced onion"],
  "kind": "ingredient"
}
```

```text
"yellow onion", "onions", "diced onion"  →  onion
"frying pan", "skillet"                  →  pan
```

The VLM prompt may include one short alias (“chopped onion”). State keys stay canonical.

If the compiler cannot map a noun onto the YOLO set or a recipe ingredient, mark the step `user_confirmation`. Do not grow vocabularies at runtime.

### Labor split

| Thing | Source |
|---|---|
| pan, knife, bowl present / bbox | YOLO |
| knife / board zone from geometry | YOLO |
| onion, garlic, rice | recipe entity list → VLM |
| chopped vs whole | attribute vocab, not a new class |
| onion in pan | location/relation, not a YOLO class |
| 1/2 tsp salt | usually `user_confirmation` |
| rare tool | VLM for that session, or ignore |

Do not ask the VLM anything YOLO already answered with high confidence.

---

## 7. YOLO26 Training and TensorRT Deployment

Train YOLO26 on the fixed V1 annotation dictionary. The recommended first deployment model is **YOLO26s**, the second-smallest standard variant, exported to a TensorRT FP16 engine.

`YOLO26s` is the better starting point than the smallest `YOLO26n`: kitchen tools are often thin, reflective, occluded, and partly outside frame. With inference only every 0.5 seconds on an RTX 3060, the accuracy retained by `s` is more valuable than maximizing throughput. Keep `YOLO26n` as a latency baseline. Only consider `YOLO26m` after a benchmark on the actual camera resolution and RTX 3060 proves it maintains the required end-to-end cadence and VRAM headroom.

Start with FP16 TensorRT. Evaluate INT8 only after an FP16 baseline exists and a representative calibration set has been collected; accept INT8 only when class-level recall, especially for knives, spoons, and tongs, remains acceptable.

### Dataset creation pipeline

Source videos are processed on a remote GPU pod. SAM 3 creates candidate segmentations for every class in the annotation dictionary; each usable mask is converted to a bounding box and emitted as a YOLO label.

```text
Source kitchen videos
  │
  ▼
Sample and deduplicate frames
  │
  ▼
Remote pod: SAM 3 prompt per dictionary class
  │
  ▼
Segmentation masks and candidate boxes
  │
  ▼
Validate / correct / reject annotations
  │
  ▼
YOLO-format dataset
  │
  ├─ local YOLO26 training
  └─ local model evaluation
```

Training data requirements:

- Split train, validation, and test sets by source video or cooking session before sampling frames. Never place near-identical frames from the same video in different splits.
- Sample frames across camera heights, lighting, cookware colors, countertop backgrounds, rotations, occlusion, motion blur, partial visibility, and empty scenes.
- Preserve both positive frames and hard negatives such as visually similar objects, packaging, towels, hands, and utensil drawers.
- Record the source video, frame timestamp, SAM 3 prompt, model version, and annotation-review status with each label.
- Audit and correct a representative sample of SAM 3 labels. Review all rare classes and failure-prone thin objects; reject low-quality masks before converting them to boxes.
- Maintain a manually reviewed, held-out test set. Auto-labeled test data only measures agreement with SAM 3, not real detection quality.

### Local training and evaluation

Train and evaluate locally against the fixed class order. Report overall mAP along with per-class precision and recall; aggregate metrics can hide a detector that misses knives or tongs.

The deployment decision is based on both accuracy and target-machine behavior:

```text
candidate: YOLO26n TensorRT FP16
candidate: YOLO26s TensorRT FP16  ← expected default
candidate: YOLO26m TensorRT FP16  ← only if benchmarked

measure: per-class recall, false positives, end-to-end latency,
   sustained cadence, GPU memory, and thermal stability
```

Export the selected checkpoint to TensorRT and run the same held-out evaluation through the TensorRT engine, not only through the training framework. A detector change or class-dictionary change produces a new versioned model artifact.

---

## 8. Kitchen State

YOLO and VLM outputs should not directly control the recipe.

Instead, both update a shared `KitchenState`. That is the only world model the recipe logic reads.

```text
YOLO  → objects (presence, bbox, coarse zone)
VLM   → residual queries (ingredient location/state, relations)
SAM 3 → later, same state
```

Example after a merge:

```json
{
  "objects": {
    "pan": {
      "present": true,
      "location": "stove",
      "source": "yolo",
      "confidence": 0.94
    },
    "knife": {
      "present": true,
      "location": "prep_area",
      "source": "yolo",
      "confidence": 0.91
    }
  },
  "ingredients": {
    "onion": {
      "state": "chopped",
      "location": "pan",
      "source": "vlm",
      "confidence": 0.86
    }
  },
  "unknown": ["salt.quantity"],
  "workspace": {
    "sink_items": 2,
    "clutter": "medium"
  }
}
```

Fusion rules:

- Prefer YOLO for closed-class object presence and boxes.
- Prefer VLM for open-class ingredients and attributes YOLO cannot provide.
- Keep `unknown` instead of guessing.
- Keep a short history of field values for hysteresis.

This keeps perception separate from application logic.

---

## 9. Recipe Compiler

The recipe compiler should support two inputs:

1. Recipe URL
2. Pasted recipe text

Pipeline:

```text
URL
 │
 ▼
Recipe extractor
 │
 ▼
Clean recipe text
 │
 ▼
Local LLM
 │
 ▼
Structured recipe JSON
```

If URL extraction fails, pasted text is the fallback.

The compiler must generate more than human-readable instructions. It should produce:

- canonical entity ids and aliases
- session entity list
- observable `expected_before` / `expected_after`
- verification type
- VLM queries derived from those expected states

If a noun cannot be mapped onto the YOLO set or a recipe ingredient, mark the step `user_confirmation`. Do not invent new classes at runtime.

Example compiled step:

```json
{
  "step": 4,
  "instruction": "Add the chopped onions to the pan.",
  "entities": [
    {
      "id": "onion",
      "aliases": ["onion", "onions", "chopped onion"],
      "kind": "ingredient"
    },
    {
      "id": "pan",
      "aliases": ["pan", "skillet", "frying pan"],
      "kind": "object"
    }
  ],
  "objects_required": ["pan"],
  "expected_before": {
    "onion": {
      "location": "prep_area",
      "state": "chopped"
    }
  },
  "expected_after": {
    "onion": {
      "location": "pan"
    }
  },
  "queries": [
    {
      "id": "onion.location",
      "entity": "onion",
      "ask": "location",
      "allowed": ["prep_area", "cutting_board", "pan", "bowl", "not_visible", "unknown"]
    }
  ],
  "verification": {
    "type": "vision"
  }
}
```

The same query machinery covers other step types:

| Step | Queries |
|---|---|
| Dice onion | `onion.state`, `onion.location` |
| Add onion to pan | `onion.location` |
| Melt butter | `butter.state` |
| Cleanup | leftover tool locations |

---

## 10. Verification Types

Not every cooking instruction can be verified visually.

The compiler should support several verification methods.

### Vision

Example:

```text
Move chopped onion from cutting board to pan.
```

Verified by comparing `KitchenState` to `expected_after`.

### Timer

Example:

```text
Simmer for 10 minutes.
```

```json
{
  "verification": {
    "type": "timer",
    "seconds": 600
  }
}
```

### User Confirmation

Example:

```text
Add 1/2 teaspoon of salt.
```

Exact quantity may not be reliably visible from the overhead camera.

```json
{
  "verification": {
    "type": "user_confirmation"
  }
}
```

### Unobservable / unknown

If required `expected_after` fields come back as `unknown`, do not treat the step as visually complete. Fall back to `user_confirmation`.

### Vision + User

The system can suggest completion based on visual evidence and ask for confirmation when confidence is low.

---

## 11. Recipe State Machine

At runtime, the system should primarily consider:

```text
previous step
current step
next step
```

Example:

```text
STEP 3
Dice onion
   ↓
STEP 4
Add onion to pan
   ↓
STEP 5
Cook for 5 minutes
```

The VLM only receives queries for the current step (and optionally the next step). It does not receive “is the step complete?”

Completion is a comparison against `KitchenState`, not a VLM field:

```text
not started    expected_before still true
in progress    neither before nor after matches
complete       expected_after matches, with enough votes
unobservable   required fields are unknown → user_confirmation
```

Example:

```text
expected_after.onion.location == kitchen_state.ingredients.onion.location
AND confidence >= 0.7
AND 2 of last 3 observations agree
```

To prevent unstable step changes, do not advance based on one observation.

Example rule:

```text
Advance if at least 2 of the last 3 KitchenState snapshots
satisfy expected_after.
```

---

## 12. Cleanup Assistant

Cleanup suggestions should use the same `KitchenState`.

Example:

```text
Current step:
Simmer for 10 minutes.

Visible:
knife
cutting board
empty onion package

Future steps:
knife not required
cutting board not required
```

The assistant can infer:

```text
unused_objects = visible_objects - future_required_objects
```

and suggest:

> You will not need the knife or cutting board anymore. This is a good time to clean them while the food simmers.

Cleanup queries can reuse the same observer API: leftover YOLO objects plus optional VLM checks for empty packages or dirty boards. The first version can use simple rules rather than an LLM for cleanup reasoning.

---

## 13. SAM 3 — Later Stage

SAM 3 should be added after the basic system works.

Potential uses:

### Ingredient Segmentation

Prompt SAM 3 with the compiled session entity list, not a global food vocabulary:

```text
carrot
chopped onion
tomato pieces
```

### Ingredient Tracking

Track ingredients as they move between:

```text
prep area → bowl → pan
```

### Pan Contents

Segment food inside the pan.

### Workspace Cleanup

Detect or segment:

- food scraps
- spilled liquid
- dirty regions
- clutter

SAM 3 should be event-driven rather than running continuously.

---

## 14. Initial Processing Schedule

Recommended MVP:

```text
Camera:
30 FPS capture
latest frame stored

Every 3 seconds:
    1. Get latest frame
    2. Run YOLO on the fixed kitchen-object set
    3. If the scene has not changed, skip the VLM
    4. Compile residual queries from the current step
    5. Skip queries YOLO already answered
    6. Run local VLM → answers[]
    7. Merge into KitchenState
    8. Compare KitchenState to expected_after
    9. Update instruction / cleanup suggestion
```

Effective perception rate:

```text
~0.33 Hz
```

This is intentionally slow but sufficient for validating the concept.

---

## 15. Suggested Project Structure

```text
kitchen-assistant/
│
├── app/
│   └── main.py
│
├── camera/
│   ├── camera.py
│   └── frame_buffer.py
│
├── perception/
│   ├── perception_manager.py
│   ├── yolo_detector.py
│   ├── vlm_client.py
│   ├── query_builder.py
│   └── sam3_segmenter.py
│
├── recipe/
│   ├── extractor.py
│   ├── compiler.py
│   ├── recipe_schema.py
│   ├── vocabularies.py
│   └── state_machine.py
│
├── state/
│   ├── kitchen_state.py
│   └── state_history.py
│
├── assistant/
│   ├── cooking_assistant.py
│   └── cleanup_assistant.py
│
├── ui/
│
├── recipes/
│   ├── source/
│   └── compiled/
│
└── config.yaml
```

Example configuration:

```yaml
camera:
  device: 0
  fps: 30

inference:
  mode: periodic
  interval_seconds: 3

vlm:
  provider: ollama
  model: TBD
  max_queries: 6

yolo:
  enabled: true
  interval_seconds: 0.5
  model: yolo26s
  engine: models/yolo26s-kitchen-v1-fp16.engine
  precision: fp16
  classes:
    - pan
    - pot
    - wok
    - lid
    - knife
    - cutting_board
    - bowl
    - plate
    - spatula
    - spoon
    - tongs
    - cup
    - bottle
    - jar
    - sponge

sam3:
  enabled: false
```

---

## 16. Development Roadmap

### V1 — Camera + Manual Observe
Capture a frame, run a small compiled query list, and return `answers[]`. No `step_completed`.

### V2 — Periodic Observe
Analyze the newest frame every three seconds with the same observer API.

### V3 — Recipe Compiler
Convert pasted recipe text into entities, `expected_before` / `expected_after`, verification type, and queries.

### V4 — Recipe State Machine
Advance only when `KitchenState` matches `expected_after` with hysteresis.

### V5 — YOLO Training and Integration
Create a SAM 3-assisted, human-reviewed dataset for the fixed dictionary. Train and evaluate YOLO26s locally, export it to TensorRT FP16, then use it as the fixed kitchen-object detector and scene-change gate. Skip VLM queries YOLO already answered.

### V6 — State Fusion
Merge YOLO and VLM results into `KitchenState` and use state history to reduce instability.

### V7 — Recipe URL Import
Add URL extraction with pasted-text fallback.

### V8 — Cleanup Assistant
Suggest cleanup actions during inactive cooking periods.

### V9 — SAM 3
Add dynamic segmentation and tracking where it provides clear value. Prompt it with the compiled session entity list.

---

## 17. MVP Success Criterion

The first meaningful demo should be:

```text
Paste recipe
      ↓
Compile recipe locally
      ↓
Start overhead camera
      ↓
Analyze one frame every 3 seconds
      ↓
YOLO (fixed kitchen objects) + VLM (compiled queries)
      ↓
Build KitchenState
      ↓
Compare KitchenState to expected_after
      ↓
Advance to next instruction automatically
```

The main hypothesis to validate is:

> Can a cooking recipe be compiled into canonical entities and visually observable states, then executed against a real kitchen scene by comparing `KitchenState` to `expected_after` using local models?
