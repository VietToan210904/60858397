"""Encode explicitly authored visual judgements; never infer scores or observations."""
import json
from datetime import datetime, timezone
from pathlib import Path

base = Path('results/ai_visual_assessment')
notes = json.loads((base / 'root_authored_notes.json').read_text(encoding='utf-8'))
for batch, entries in notes.items():
    rows = [json.loads(line) for line in (base / 'templates' / f'generation_{batch}.template.jsonl').read_text().splitlines()]
    assert len(rows) == len(entries) == 12
    for row, (verdicts, scores, usable, observation, uncertainty, errors) in zip(rows, entries):
        assert len(verdicts) == len(row['criteria'])
        row.update(assessor_id='codex_root_visual_review', assessor_model='not_exposed',
                   assessed_at_utc=datetime.now(timezone.utc).isoformat(), review_status='assessed',
                   ai_usable_concept=usable, confidence='medium' if uncertainty else 'high',
                   uncertainty_reasons=uncertainty, error_categories=errors or ['none_observed'],
                   observations=[{'asset':'image','description':observation,'region':None}])
        row['scores'] = dict(zip(row['scores'], scores))
        for criterion, verdict in zip(row['criteria'], verdicts):
            criterion.update(verdict={'M':'met','A':'absent','U':'uncertain'}[verdict], evidence=observation)
    target = base / 'assessments' / f'generation_{batch}.jsonl'
    if target.exists():
        old = [json.loads(line) for line in target.read_text().splitlines()]
        for previous, current in zip(old, rows):
            current['assessed_at_utc'] = previous['assessed_at_utc']
        assert old == rows, f'Existing assessment changed: {target}'
    else:
        target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
print(f'Encoded {sum(map(len,notes.values()))} actual image assessments.')
