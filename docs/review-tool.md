# Kitchen Lab review tool

Open https://win-ckecmhj83p0.tailc1e2f.ts.net:8877/ from a device on your Tailscale
network. Locally, use http://127.0.0.1:8877/. The Windows host must be awake and
connected. The backend binds to loopback; Tailscale Serve provides private HTTPS.
Existing Tailscale services on ports 80 and 8765 are left intact.

## Review workflow

- **Compare models:** all 2,187 one-second samples across 15 episodes. Select a
  video, scrub the timeline, jump to a second, or play sampled frames. Toggle
  full frame/cooking area, class filters, confidence and original SAM3 masks.
  Both models use original-frame coordinates. YOLO only predicts inside
  `[500,330,1120,1080]`; the rectangle marks that scope in full-frame view.
- **Correct labels:** choose full-frame or crop scope; initialize from visible
  SAM3/YOLO boxes, original training labels, or an empty draft. Confidence, class
  and view filters affect the "visible" source buttons. Drag to add, move or
  resize boxes, change classes, delete duplicates, and record a note/decision.
  Save explicitly. Approving an empty frame marks an explicit negative.
- **Training run:** inspect the 60-epoch loss curves, exact configuration, log,
  checkpoint and class coverage. Click any of the 45 actual training samples
  to inspect its labels. These original labels were assistant-reviewed, not
  independently verified. Training-set diagnostics are not test accuracy.
- **Export reviews:** downloads a ZIP containing every saved review decision,
  and paired images/YOLO labels only for approved frames. Crop reviews export
  cropped images with crop-normalized coordinates; full reviews export original
  images with full-frame-normalized coordinates. Preserve session grouping in
  future train/test splits. This tool does not start retraining automatically.

Edits are box corrections. SAM3 masks can be inspected but are not edited or
regenerated. Original annotation and training files are never overwritten.

## Storage and reproducibility

Review decisions and revision history persist in
`outputs/review-tool/reviews.sqlite3` (SQLite WAL mode). Concurrent stale saves
are rejected, and unsaved changes trigger a navigation warning. Back up using
SQLite's backup API or export reviews; do not copy a live SQLite DB without its
WAL. Review data, predictions, model weights and raw media are git-ignored.

All-frame YOLO predictions are cached under `outputs/review-tool/predictions`.
Each records image and model SHA-256, crop, inference timing and a confidence
floor of 0.05. The UI initially shows detections at 0.25. Regenerate the cache
after replacing weights using `.venv-training/Scripts/python.exe
scripts/cache_review_predictions.py`; the cache invalidates changed model hashes.

Start/restart with `powershell -File scripts/start_review.ps1`. The backend has
no external web dependencies and uses the existing training environment's Pillow.
Run focused tests with `.venv-training/Scripts/python.exe scripts/test_review_server.py`.
The Windows scheduled task **Kitchen Lab Review** starts it again at user login.

Tailscale route: `tailscale serve --bg --https=8877 http://127.0.0.1:8877`.
Remove only this route with `tailscale serve --https=8877 off`.
Tailnet policy controls which devices can reach the service. Any authorized
visitor can review/export this dataset. The app rejects cross-origin writes
and serves only allowlisted artifacts, not arbitrary workspace paths.
