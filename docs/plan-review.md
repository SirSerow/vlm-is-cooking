# Plan feasibility review — 2026-09-05

Verdict: feasible as a constrained, user-supervised prototype. The architecture
separating observation from recipe progression is sensible. Automatic execution
of arbitrary recipes is an unproven research goal. None of the issues below block
preparing an offline SAM3 annotation environment.

## Changes needed before implementation

1. **Class IDs:** the example YAML reorders classes after knife, conflicting with
   the annotation table. Corrected the example and added a machine-readable
   dictionary. Training, export and runtime must consume the same version.
2. **Scene gating:** stationary tool boxes do not imply unchanged ingredients.
   Use image/region changes plus a maximum VLM refresh interval, and force an
   observation on step changes. Start with unconditional periodic observations.
3. **Fresh evidence:** store frame ID, capture timestamp, field timestamp, source
   and expiration. Two snapshots reusing the same VLM answer are not two votes.
   Count distinct fresh observations collected after step entry; handle stale or
   contradictory evidence explicitly. Matching an already-true after-state must
   not automatically skip a new step that requires an action or transition.
4. **Multiple instances and portions:** a dictionary keyed by `pan` or `onion`
   cannot represent two pans or split ingredients. Use instance IDs with class
   attributes and relations to specific containers. For the first demo explicitly
   restrict recipes to one relevant instance/portion at a time.
5. **Attribute structure:** `chopped`, `raw`, and `cooking` are not mutually
   exclusive. Separate preparation shape, cooking state, and visibility. A lid
   or occlusion is unknown, not evidence of absence. Model-reported confidence
   is not calibrated; validate thresholds on manually labelled observations.
6. **Annotation quality:** review missed objects, not only predicted masks.
   Disambiguate pan/pot/wok and bottle/jar; fix duplicate predictions and missing
   handles. YOLO has no generic ignore label: withhold unresolved frames. Keep
   the held-out test set fully human-reviewed and independent by session.
7. **Scheduling and hardware:** 3 seconds is a target, not an established RTX
   3060 VLM latency. Benchmark the chosen quantized VLM alongside YOLO, use a
   single in-flight observation and drop superseded frames. Build/validate the
   TensorRT engine on the deployment platform. Do not promise simultaneous
   training and VLM serving within the same VRAM budget.
8. **Verification semantics:** specify timer start/pause/reset and manual undo,
   skip and confirmation. Time elapsed or appearance alone cannot establish
   temperature or food safety. Cleanup suggestions need explicit prerequisites,
   including whether the tool is still in use. Put state fusion before automatic
   progression in the roadmap; offline SAM3 annotation is independent of V9's
   runtime segmentation.

## Suggested order

Prepare SAM3 → small annotation pilot → human-reviewed dataset tooling → detector
baseline. In parallel in the project schedule, validate manual VLM observations
on a few simple recipes before building a general compiler. Then implement typed
state with freshness, periodic observation, and supervised step progression.

Suggested pilot: 100–200 diverse frames from several sessions, including empty
scenes and thin tools. Agree on class-level recall and review-cost acceptance
criteria after measuring that pilot, before paying for a full video corpus run.

## Verified upstream information

- [Meta SAM3](https://github.com/facebookresearch/sam3) supports text-prompted
  image/video segmentation; checkpoint access is gated. The environment uses
  its documented current PyTorch/CUDA stack and pins the source revision.
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) exists and
  supports the proposed detector family. Published benchmark results do not
  establish this application's RTX 3060 latency or kitchen accuracy.
