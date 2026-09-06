# Kitchen Lab review tool

Open https://win-ckecmhj83p0.tailc1e2f.ts.net:8877/ from a device on your Tailscale
network. Locally, use http://127.0.0.1:8877/. The Windows host must be awake and
connected. The backend binds to loopback; Tailscale Serve provides private HTTPS.
Existing Tailscale services on ports 80 and 8765 are left intact.

## Quick start on this Windows machine

1. Open **PowerShell**.
2. Run:

   ```powershell
   Set-Location 'S:\Projects\vlm-is-cooking'
   powershell -File .\scripts\start_review.ps1
   ```

3. Open **[Kitchen Lab locally](http://127.0.0.1:8877/)** in your browser.

The launcher starts the server in the background, so you can close PowerShell
after launching it. Running the launcher again leaves an already-running server
in place; it does not restart it. The scheduled task **Kitchen Lab Review** also
starts the server when you log into Windows.

**Do not open `review-app/index.html` directly.** The app needs its HTTP server
to load scripts, images, annotations and saved reviews. A `file://` URL will not
work correctly. No Runpod pod or new inference job is needed to review the
already-cached results.

## Open it from another device

Connect that device to your Tailscale network, then open:

**[Kitchen Lab through Tailscale](https://win-ckecmhj83p0.tailc1e2f.ts.net:8877/)**

Keep this Windows machine awake, connected to Tailscale, and running the server.
The Tailscale address is private; it is not a public website. On another device,
`127.0.0.1` refers to that device, so use the Tailscale link instead.

## If the page does not load

From the project folder, check the local server:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8877/api/status'
```

A JSON status response confirms the server is reachable. `state: complete`
describes the prediction cache, not whether your label reviews are finished.

If the request fails, run the launcher above and check its logs:

```powershell
Get-Content .\outputs\review-server.log -Tail 30
Get-Content .\outputs\review-server-error.log -Tail 50
```

Normal HTTP request lines also appear in the error log; look for a Python
traceback or an explicit error. For direct startup diagnostics, run the server
in the foreground **only when another instance is not already running**:

```powershell
& .\.venv-training\Scripts\python.exe .\scripts\review_server.py
```

Keep this terminal open. Press **Ctrl+C** to stop this foreground instance.
An “address already in use” error means another process occupies port 8877;
do not start multiple instances on that port.

If the local link works but the Tailscale link fails, check the Windows host's
Tailscale connection and route:

```powershell
tailscale status
tailscale serve status
```

The expected route proxies HTTPS port 8877 to `http://127.0.0.1:8877`.
If it is missing, the route setup command is listed below under storage and
reproducibility. If the page loads but displays old assets after an update,
save any pending edits, then hard-refresh with **Ctrl+F5**.

## Requirements and moving to another machine

The instructions above assume the existing local installation. A Git clone
alone is not enough: the app reads local artifacts that are excluded from Git.
It needs:

- `.venv-training` with Python and Pillow; see
  [the training guide](train-guide.md#1-check-the-existing-environment)
  for environment recreation instructions.
- `outputs/all-videos-1fps`, including the manifest, source images and annotations.
- `config/kitchen_classes.v1.json` and `data/curated-crops-v1/provenance.json`.
- `outputs/local-training/yolo26s-curated-crops-v1` and
  `outputs/local-training.log` for the training page.
- `outputs/review-tool/predictions` for cached YOLO comparisons and
  `models/yolo26s-cooking-crops-v1.pt` for checkpoint download.
- Your existing review database if you want to retain saved decisions; see the
  backup instructions below.

The provided Tailscale hostname and Windows login task belong to this machine.
They are not installed automatically by cloning the repository elsewhere.

## Review workflow

- **Compare models:** all 2,187 one-second samples across 15 episodes. Select a
  video, scrub the timeline, jump to a second, or play sampled frames. Toggle
  full frame/cooking area, class filters, confidence and original SAM3 masks.
  Both models use original-frame coordinates. YOLO only predicts inside
  `[500,330,1120,1080]`; the rectangle marks that scope in full-frame view.
- **Correct labels:** current dataset boxes load automatically when there is no
  saved review. Saved reviews, including empty ones, take precedence. All source
  boxes in the chosen scope load, regardless of comparison confidence/class filters.
  Choose full-frame or crop scope; optionally initialize from visible
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

After correcting a frame, select **Approved within scope**, click **Save review**,
then click **Replace original dataset labels**. The confirmation applies only to
that frame, not every reviewed frame. It updates the original annotation JSON and
`candidate_labels` YOLO text file, and refreshes the app's dataset preview.

For crop reviews, source boxes intersecting the cooking area are replaced;
boxes completely outside it remain unchanged. An unchanged box that extends
beyond the crop retains its original full-frame extent and mask. A uniquely
matched unchanged box retains its mask even if its class changes. New, resized
or ambiguously matched boxes have no segmentation mask; masks are not fabricated.
Approving an empty scope and applying it removes source boxes in that scope.

Each replacement saves original files and review metadata under
`outputs/review-tool/source-backups`. Stale reviews or changed source annotations
are rejected. Save alone still stores a separate review. Existing training
datasets, checkpoints, cached YOLO predictions, static SAM3 previews and old
summary reports are not rebuilt by replacement. Use a fresh review export for
training on your latest approved corrections.

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

Start with `powershell -File scripts/start_review.ps1`. The backend has
no external web dependencies and uses the existing training environment's Pillow.
Run focused tests with `.venv-training/Scripts/python.exe scripts/test_review_server.py`.
The Windows scheduled task **Kitchen Lab Review** starts it again at user login.

Tailscale route: `tailscale serve --bg --https=8877 http://127.0.0.1:8877`.
Remove only this route with `tailscale serve --https=8877 off`.
Tailnet policy controls which devices can reach the service. Any authorized
visitor can review/export this dataset. The app rejects cross-origin writes
and serves only allowlisted artifacts, not arbitrary workspace paths.
