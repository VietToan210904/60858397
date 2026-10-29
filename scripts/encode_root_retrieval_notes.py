"""Serialize manually authored photo annotations without inferring product traits."""
import json
from datetime import datetime, timezone
from pathlib import Path

base = Path('results/retrieval/ai_assessment/v1')
notes = json.loads((base/'annotations/root_authored_visual_notes.json').read_text(encoding='utf-8'))
template = {r['annotation_index']:r for r in map(json.loads,(base/'annotation_template.jsonl').read_text(encoding='utf-8').splitlines())}
evidence = {r['annotation_index']:r for r in map(json.loads,(base/'product_evidence.jsonl').read_text(encoding='utf-8').splitlines())}
rows=[]
for entry in notes:
    index, types, scores, note, *support = entry
    row=template[index]
    assert len(scores)==len(row['traits'])==10
    uncertain=[key for field in support if field.startswith('uncertain:') for key in field.split(':',1)[1].split(',')]
    row.update(matched_types=types.split(',') if types else [], uncertain_types=uncertain,
               traits=dict(zip(row['traits'],[None if c=='?' else int(c) for c in scores])),
               explicit_lightweight_source=None, confidence='medium',visual_notes=note,
               assessor_id='codex_root_visual_review',assessor_model='not_exposed',
               assessed_at_utc=datetime.now(timezone.utc).isoformat())
    row['source_support']=[{'field':field,'exact_source_values':evidence[index]['source_facts'][field]} for field in support if not field.startswith('uncertain:')]
    rows.append(row)
target=base/'annotations/root-01-08.jsonl'
if target.exists():
    old={r['annotation_index']:r for r in map(json.loads,target.read_text(encoding='utf-8').splitlines())}
    for row in rows:
        if row['annotation_index'] in old:
            row['assessed_at_utc']=old[row['annotation_index']]['assessed_at_utc']
            assert old[row['annotation_index']]==row, 'Existing visual judgement changed.'
target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
print(f'Encoded {len(rows)} actual photo annotations.')
