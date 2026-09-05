#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vlm-is-cooking
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/workspace/venvs/sam3/bin/python
mkdir -p outputs/all-videos-1fps
pids=()
for shard in 0 1 2 3; do
  "$PY" -u scripts/sam3_batch.py data/annotation-frames outputs/all-videos-1fps \
    --checkpoint /workspace/models/sam3/sam3.pt --shards 4 --shard-index "$shard" \
    > "outputs/all-videos-1fps/worker-$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if ((failed)); then
  echo 'At least one annotation worker failed' > outputs/all-videos-1fps/FAILED
  exit 1
fi
"$PY" scripts/summarize_annotations.py data/annotation-frames outputs/all-videos-1fps \
  > outputs/all-videos-1fps/validation.log
cp data/annotation-frames/manifest.json outputs/all-videos-1fps/manifest.json
cp config/kitchen_classes.v1.json outputs/all-videos-1fps/classes.json
cp data/ANNOTATION-README.md outputs/all-videos-1fps/README.md
cp /workspace/venvs/sam3/installed.freeze.txt outputs/all-videos-1fps/installed.freeze.txt
touch outputs/all-videos-1fps/COMPLETE
tar -cf outputs/all-videos-1fps.tar -C outputs all-videos-1fps
sha256sum outputs/all-videos-1fps.tar > outputs/all-videos-1fps.tar.sha256
