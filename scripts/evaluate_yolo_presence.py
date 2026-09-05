"""Score assistant visual class-presence judgments; deliberately not detector mAP."""
import json
import html
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/yolo-self-evaluation'


def prepare_sample():
    frames=json.loads((ROOT/'outputs/all-videos-1fps/manifest.json').read_text())
    used={r['image'] for r in json.loads((ROOT/'data/curated-crops-v1/provenance.json').read_text())}
    sample=[]
    for video in sorted({f['source_video'] for f in frames}):
        pool=[(i,f) for i,f in enumerate(frames) if f['source_video']==video and f['image'] not in used]
        n=min(10,len(pool))
        for slot in range(n):
            i,f=pool[round((slot+.5)*len(pool)/n-.5)]
            sample.append(dict(id=i,slot=slot,**f))
    ref=json.loads((ROOT/'config/yolo_self_evaluation.v1.json').read_text())
    assert [f['id'] for f in sample]==ref['sampled_frame_ids'], 'Sample changed; visual references must be reviewed again.'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'sample.json').write_text(json.dumps(sample,indent=2))


def evaluate(threshold):
    ref=json.loads((ROOT/'config/yolo_self_evaluation.v1.json').read_text())
    sample=json.loads((OUT/'sample.json').read_text())
    classes=json.loads((ROOT/'config/kitchen_classes.v1.json').read_text())['classes']
    per=[dict(name=c['name'],tp=0,fp=0,fn=0,tn=0,extra_boxes=0) for c in classes]
    rows=[]
    for frame in sample:
        fid=frame['id']
        if str(fid) in ref['excluded_frames']:continue
        truth=set(ref['classes_by_video_slot'][frame['source_video'][:3]][frame['slot']])
        ignored=set(ref['ignore_classes_by_frame'].get(str(fid),[]))
        truth-=ignored
        prediction=json.loads((ROOT/'outputs/review-tool/predictions'/f'{fid:05d}.json').read_text())
        assert prediction['image_sha256']==frame['image_sha256'], 'Prediction image provenance mismatch'
        counts=Counter(b['class_id'] for b in prediction['boxes'] if b['score']>=threshold and b['class_id'] not in ignored)
        predicted=set(counts)
        extra=sum(max(0,n-1) for n in counts.values())
        rows.append(dict(id=fid,video=frame['source_video'],timestamp=frame['timestamp_seconds'],truth=sorted(truth),
                         predicted=sorted(predicted),missing=sorted(truth-predicted),unexpected=sorted(predicted-truth),
                         extra_same_class_boxes=extra,ignored=sorted(ignored),exact_class_set=truth==predicted))
        for c in range(len(classes)):
            if c in ignored:continue
            per[c]['tp' if c in truth and c in predicted else 'fn' if c in truth else 'fp' if c in predicted else 'tn']+=1
            per[c]['extra_boxes']+=max(0,counts[c]-1)
    def metrics(r):
        tp,fp,fn=r['tp'],r['fp'],r['fn']
        r.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                 f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,support=tp+fn)
        return r
    for p in per:metrics(p)
    totals=metrics({k:sum(p[k] for p in per) for k in ('tp','fp','fn','tn','extra_boxes')})
    totals.update(frames=len(rows),exact_class_set_frames=sum(r['exact_class_set'] for r in rows),
                  frames_with_extra_boxes=sum(r['extra_same_class_boxes']>0 for r in rows),
                  reference_empty_frames=sum(not r['truth'] for r in rows),
                  clean_empty_frames=sum(not r['truth'] and not r['predicted'] for r in rows))
    return dict(threshold=threshold,totals=totals,per_class=per,frames=rows)


def main():
    prepare_sample()
    main=evaluate(.25)
    main['threshold_sensitivity']=[dict(threshold=t,**evaluate(t)['totals']) for t in (.1,.25,.4,.5,.7)]
    main['method']='Assistant-audited class presence per frame. A class counts as detected if ANY prediction has that class; location and duplicate boxes do not affect presence precision. Not mAP, not box-level precision.'
    main['limitations']=['Same recording session as training; correlated frames, not an independent test.',
                        'Assistant visual labels, not human-verified ground truth.',
                        '143 inspected frames; one excluded for motion/ambiguous content; 142 scored.',
                        'Four class-frame decisions ignored due to ambiguous cropped containers.',
                        'Video-stratified sample is not weighted by video duration.',
                        'Confidence sensitivity reuses this sample; no threshold was selected for deployment.']
    (OUT/'metrics.json').write_text(json.dumps(main,indent=2))
    print(json.dumps(main['totals'],indent=2))
    for p in main['per_class']:print(p['name'],p['support'],p['tp'],p['fp'],p['fn'],p['precision'],p['recall'],p['extra_boxes'])
    print('WORST',[(r['id'],r['missing'],r['unexpected'],r['extra_same_class_boxes']) for r in sorted(main['frames'],key=lambda r:len(r['missing'])+len(r['unexpected'])+r['extra_same_class_boxes'],reverse=True)[:12]])
    publish(main)


def publish(result):
    t=result['totals']
    pct=lambda v:'—' if v is None else f'{v:.1%}'
    names=[p['name'] for p in result['per_class']]
    rows=[p for p in result['per_class'] if p['support'] or p['fp']]
    md=['# YOLO assistant evaluation — 2026-09-05','',
        '**Conclusion: not reliable enough for automatic annotation.** The main errors are missing objects and duplicate boxes.','',
        'I visually inspected 143 cooking-area crops from all 15 videos, excluding the 45 training frames. '
        'One motion-blurred/ambiguous frame was excluded, leaving 142 scored frames. '
        'Four ambiguous class/frame decisions were ignored. Video names supplied context; visible image content determined labels.','',
        'References were recorded from original image contact sheets before comparison with cached predictions. '
        'Earlier predictions from this session had already been seen, so this is not a fully blinded review.','',
        '## Results at confidence 0.25','',
        f'- Class-presence recall: **{pct(t["recall"])}** ({t["tp"]}/{t["support"]}); {t["fn"]} missed class occurrences.',
        f'- Exact visible class set: **{t["exact_class_set_frames"]}/{t["frames"]} ({t["exact_class_set_frames"]/t["frames"]:.1%})**.',
        '- Correct class set with no extra same-class boxes: **89/142 (62.7%)**; this still does not check localization.',
        f'- Extra same-class boxes: **{t["extra_boxes"]} across {t["frames_with_extra_boxes"]} frames**.',
        f'- Empty crops: **{t["clean_empty_frames"]}/{t["reference_empty_frames"]}** correctly produced no predictions.',
        f'- Class-presence precision: **{pct(t["precision"])}** ({t["tp"]}/{t["tp"]+t["fp"]}). '
        'This forgiving metric counts a class as correct whenever it is present anywhere in the crop; '
        'it ignores box location and collapses duplicates. It is NOT 99.5% annotation accuracy.','',
        '| Class | Visible frames | Detected | Missed | Presence recall |',
        '|---|---:|---:|---:|---:|']
    md += [f'| {p["name"]} | {p["support"]} | {p["tp"]} | {p["fn"]} | {pct(p["recall"])} |' for p in rows]
    md += ['', 'Lid and bowl results have small support. Spoon and tongs results have only one and two examples. '
           'Pot, wok, cup, bottle, jar and sponge have no confidently scored positives; their recall cannot be estimated.','',
           '## Concrete failures','',
           '- Frame 143, cutting onion: plate and bowl detected; visible knife and cutting board missed.',
           '- Frame 1150, adding cut onion: visible pan and spatula both missed.',
           '- Frame 1420, adding chicken: tongs classified as spatula.',
           '- Frame 1973, opening lid: visible pan and spoon both missed.',
           '- Frame 1911, closing lid: duplicate pan and lid boxes.','',
           '## Confidence sensitivity','',
           '| Confidence | Presence precision | Presence recall | Frames with extra boxes |',
           '|---:|---:|---:|---:|']
    md += [f'| {r["threshold"]:.2f} | {pct(r["precision"])} | {pct(r["recall"])} | {r["frames_with_extra_boxes"]} |' for r in result['threshold_sensitivity']]
    md += ['', 'Lowering confidence to 0.10 raises recall to 86.0%, but extra boxes appear in 56 frames instead of 17. '
           'Threshold changes alone do not solve the problem. These thresholds were inspected on the same sample, not independently validated.','',
           '## Limits and next step','',
           'This measures per-frame object-class presence in the trained crop, not box IoU, detector mAP, mask quality, '
           'action recognition or full-frame performance. Frames come from the same session as training and are correlated. '
           'Sampling is approximately balanced across videos, not proportional to duration. Assistant judgments are not human-verified ground truth.','',
           'Prioritize corrected examples of knives, cutting boards, spatulas, tongs, spoons and food-filled pans; '
           'include occlusion and tool motion. Keep a separately recorded session for final evaluation. '
           'Do not turn these presence-only judgments directly into box training labels.','',
           'Reproducibility: `config/yolo_self_evaluation.v1.json` contains the visual judgments; '
           '`outputs/yolo-self-evaluation/sample.json` contains exact image IDs/hashes; '
           '`outputs/yolo-self-evaluation/metrics.json` contains all per-frame outcomes. '
           'Run `scripts/evaluate_yolo_presence.py` in the training environment to recompute.']
    (ROOT/'docs/yolo-self-evaluation.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    style='body{background:#101416;color:#e7efec;font:15px/1.6 Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px}a{color:#c4f580}h1{font-weight:500}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{text-align:left;padding:9px;border-bottom:1px solid #354044}.notice{padding:16px;background:#30291d;color:#e4ca93;border-radius:8px}small{color:#a0b2b8}details{background:#191f22;margin:10px 0;padding:12px;border-radius:8px}summary{cursor:pointer}img{width:160px;border-radius:5px}pre{white-space:pre-wrap;font:13px/1.65 Segoe UI,sans-serif}'
    page=[f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>YOLO visual audit · Kitchen Lab</title><style>{style}</style><body>',
          '<a href="/">← Kitchen Lab</a><h1>YOLO visual audit</h1><p>143 crops inspected · 142 scored · 15 videos · confidence 0.25</p>',
          '<p class="notice">Experimental same-session evaluation by the assistant. Object presence only — not box accuracy or independently verified ground truth.</p>',
          f'<h2>{pct(t["recall"])} presence recall · 100/142 complete class sets</h2>',
          '<p>49 missed class occurrences. 18 extra same-class boxes across 17 frames. Strongest gaps: knives, cutting boards and utensils.</p>',
          '<table><tr><th>Class</th><th>Visible</th><th>Detected</th><th>Missed</th><th>Recall</th></tr>']
    page += [f'<tr><td>{p["name"]}</td><td>{p["support"]}</td><td>{p["tp"]}</td><td>{p["fn"]}</td><td>{pct(p["recall"])}</td></tr>' for p in rows]
    page += ['</table><h2>Inspect every judgment</h2><p>Open a frame to compare its predictions. “Extra” counts additional same-class boxes; localization is not scored.</p>']
    for video in sorted({r['video'] for r in result['frames']}):
        frames=[r for r in result['frames'] if r['video']==video]
        page.append(f'<details><summary>{html.escape(video)} · {sum(r["exact_class_set"] for r in frames)}/{len(frames)} exact class sets</summary><table><tr><th>Frame</th><th>Visible judgment</th><th>Missed</th><th>Unexpected / extra</th></tr>')
        for r in frames:
            label=lambda ids:', '.join(names[c] for c in ids) or 'none'
            page.append(f'<tr><td><a href="/?frame={r["id"]}"><img loading="lazy" src="/thumb?id={r["id"]}" alt="Frame {r["id"]}"><br>Frame {r["id"]} · {r["timestamp"]:.1f}s</a></td><td>{label(r["truth"])}</td><td>{label(r["missing"])}</td><td>{label(r["unexpected"])} / {r["extra_same_class_boxes"]} extra</td></tr>')
        page.append('</table></details>')
    page += ['<details><summary>Full methodology, results and limitations</summary><pre>'+html.escape('\n'.join(md))+'</pre></details></body></html>']
    (ROOT/'review-app/evaluation.html').write_text('\n'.join(page), encoding='utf-8')


if __name__=='__main__':main()
