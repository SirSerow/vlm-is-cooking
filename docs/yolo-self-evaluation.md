# YOLO assistant evaluation — 2026-09-05

**Conclusion: not reliable enough for automatic annotation.** The main errors are missing objects and duplicate boxes.

I visually inspected 143 cooking-area crops from all 15 videos, excluding the 45 training frames. One motion-blurred/ambiguous frame was excluded, leaving 142 scored frames. Four ambiguous class/frame decisions were ignored. Video names supplied context; visible image content determined labels.

References were recorded from original image contact sheets before comparison with cached predictions. Earlier predictions from this session had already been seen, so this is not a fully blinded review.

## Results at confidence 0.25

- Class-presence recall: **79.2%** (187/236); 49 missed class occurrences.
- Exact visible class set: **100/142 (70.4%)**.
- Correct class set with no extra same-class boxes: **89/142 (62.7%)**; this still does not check localization.
- Extra same-class boxes: **18 across 17 frames**.
- Empty crops: **8/8** correctly produced no predictions.
- Class-presence precision: **99.5%** (187/188). This forgiving metric counts a class as correct whenever it is present anywhere in the crop; it ignores box location and collapses duplicates. It is NOT 99.5% annotation accuracy.

| Class | Visible frames | Detected | Missed | Presence recall |
|---|---:|---:|---:|---:|
| pan | 84 | 75 | 9 | 89.3% |
| lid | 19 | 19 | 0 | 100.0% |
| knife | 32 | 20 | 12 | 62.5% |
| cutting_board | 20 | 10 | 10 | 50.0% |
| bowl | 7 | 6 | 1 | 85.7% |
| plate | 49 | 45 | 4 | 91.8% |
| spatula | 22 | 12 | 10 | 54.5% |
| spoon | 1 | 0 | 1 | 0.0% |
| tongs | 2 | 0 | 2 | 0.0% |

Lid and bowl results have small support. Spoon and tongs results have only one and two examples. Pot, wok, cup, bottle, jar and sponge have no confidently scored positives; their recall cannot be estimated.

## Concrete failures

- Frame 143, cutting onion: plate and bowl detected; visible knife and cutting board missed.
- Frame 1150, adding cut onion: visible pan and spatula both missed.
- Frame 1420, adding chicken: tongs classified as spatula.
- Frame 1973, opening lid: visible pan and spoon both missed.
- Frame 1911, closing lid: duplicate pan and lid boxes.

## Confidence sensitivity

| Confidence | Presence precision | Presence recall | Frames with extra boxes |
|---:|---:|---:|---:|
| 0.10 | 96.2% | 86.0% | 56 |
| 0.25 | 99.5% | 79.2% | 17 |
| 0.40 | 99.4% | 70.8% | 5 |
| 0.50 | 100.0% | 64.0% | 3 |
| 0.70 | 100.0% | 43.2% | 0 |

Lowering confidence to 0.10 raises recall to 86.0%, but extra boxes appear in 56 frames instead of 17. Threshold changes alone do not solve the problem. These thresholds were inspected on the same sample, not independently validated.

## Limits and next step

This measures per-frame object-class presence in the trained crop, not box IoU, detector mAP, mask quality, action recognition or full-frame performance. Frames come from the same session as training and are correlated. Sampling is approximately balanced across videos, not proportional to duration. Assistant judgments are not human-verified ground truth.

Prioritize corrected examples of knives, cutting boards, spatulas, tongs, spoons and food-filled pans; include occlusion and tool motion. Keep a separately recorded session for final evaluation. Do not turn these presence-only judgments directly into box training labels.

Reproducibility: `config/yolo_self_evaluation.v1.json` contains the visual judgments; `outputs/yolo-self-evaluation/sample.json` contains exact image IDs/hashes; `outputs/yolo-self-evaluation/metrics.json` contains all per-frame outcomes. Run `scripts/evaluate_yolo_presence.py` in the training environment to recompute.
