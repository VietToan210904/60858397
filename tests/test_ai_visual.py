"""Synthetic review fixtures verify AI/human separation, provenance and clustering."""
import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from stylesphere.common.io import canonical_hash, sha256_file
from stylesphere.reporting.ai_visual import (GEN_SCORES, RECON_CRITERIA, aggregate, assessment_template,
    export_packets, validate_assessment, validate_export)


def synthetic_export(tmp_path, cases=(('a','p1',1),), monkeypatch=None):
    root=tmp_path
    (root/'config').mkdir()
    source=Path(__file__).resolve().parents[1]
    (root/'config/ai_visual_assessment.yaml').write_bytes((source/'config/ai_visual_assessment.yaml').read_bytes())
    (root/'docs').mkdir()
    (root/'docs/AI_VISUAL_ASSESSMENT_PROTOCOL.md').write_bytes((source/'docs/AI_VISUAL_ASSESSMENT_PROTOCOL.md').read_bytes())
    image=root/'synthetic-unit-image.png'
    Image.new('RGB',(32,32),'red').save(image)
    records=[{'pair':'generation','model':m,'case_id':p,'seed':s,'run_id':'synthetic-unit-fixture',
              'prompt':'Synthetic unit-test red image','mandatory_criteria':['red background'],
              'input_sha256':canonical_hash(p),'artifacts':{'image':image.name},'artifact_sha256':{'image':sha256_file(image)}} for m,p,s in cases]
    destination=root/'results/ai_visual_assessment'
    manifest=export_packets(root,root/'results',destination,records=records)
    cards=json.loads((destination/'public_cards.json').read_text())
    mapping=json.loads((destination/'private_mapping.json').read_text())
    return root,destination,manifest,cards,mapping


def complete(card,protocol,score=4):
    row=assessment_template(card,protocol)
    row.update(assessor_id='synthetic-unit-reviewer',assessor_model='not_exposed',assessed_at_utc='2026-01-01T00:00:00Z',
               review_status='assessed',confidence='medium',uncertainty_reasons=['Synthetic unit-test only'],
               scores={name:score for name in GEN_SCORES},ai_usable_concept=int(score>=3),
               criteria=[{'criterion':'red background','verdict':'met','evidence':'Synthetic fixture is red'}],
               observations=[{'asset':'image','description':'Synthetic fixture is red','region':None}],error_categories=['none_observed'])
    return row


def test_export_freezes_packets_preserves_human_files_and_rejects_tampering(tmp_path):
    human=tmp_path/'results/generation/human_review/ratings.csv'
    human.parent.mkdir(parents=True)
    human.write_text('review_id,usable\noriginal,\n')
    before=human.read_bytes()
    root,dest,manifest,cards,_=synthetic_export(tmp_path)
    assert human.read_bytes()==before
    assert cards[0]['assessment_id'].startswith('ai_') and 'model' not in cards[0]
    assert validate_export(dest)['selection_sha256']==manifest['selection_sha256']
    template=json.loads(next((dest/'templates').glob('*.jsonl')).read_text())
    assert template['review_status']=='pending' and all(v is None for v in template['scores'].values())
    with pytest.raises(ValueError): validate_assessment(template,cards[0],manifest['protocol_sha256'])
    display=dest/cards[0]['display_assets']['image']['path']
    Image.new('RGB',(512,512),'blue').save(display)
    with pytest.raises(ValueError,match='artifact changed'): validate_export(dest)


def test_semantic_schema_requires_bound_evidence_and_full_criterion_coverage(tmp_path):
    _,_,manifest,cards,_=synthetic_export(tmp_path)
    card=cards[0]; row=complete(card,manifest['protocol_sha256'])
    validate_assessment(row,card,manifest['protocol_sha256'])
    wrong=copy.deepcopy(row);wrong['evidence_sha256']['image']='0'*64
    with pytest.raises(ValueError,match='evidence hashes'): validate_assessment(wrong,card,manifest['protocol_sha256'])
    wrong=copy.deepcopy(row);wrong['criteria'][0]['verdict']='uncertain'
    with pytest.raises(ValueError,match='Positive AI shortlist'): validate_assessment(wrong,card,manifest['protocol_sha256'])
    wrong=copy.deepcopy(row);wrong['criteria']=[]
    with pytest.raises(ValueError,match='Every|Each'): validate_assessment(wrong,card,manifest['protocol_sha256'])
    wrong=copy.deepcopy(row);wrong['assessor_kind']='human'
    with pytest.raises(ValueError,match='AI assessor'): validate_assessment(wrong,card,manifest['protocol_sha256'])


def test_aggregation_equal_prompt_weight_retains_missing_and_rejects_duplicate_judges(tmp_path):
    cases=[('a','p1',seed) for seed in (1,2,3)]+[('a','p2',1),('a','unreviewed',1)]+[('b','p1',1),('b','p2',1)]
    _,dest,manifest,cards,mapping=synthetic_export(tmp_path,cases)
    lookup={m['assessment_id']:m for m in mapping}
    rows=[]
    for card in cards:
        item=lookup[card['assessment_id']]
        if item['case_id']=='unreviewed': continue
        score=5 if item['model']=='a' and item['case_id']=='p1' else 1
        rows.append(complete(card,manifest['protocol_sha256'],score))
    file=dest/'assessments/unit-test.jsonl'
    file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=aggregate(dest)
    model=next(m for m in result['models'] if m['model']=='a')
    assert model['metrics']['ai_prompt_adherence']=={'mean':3.,'n_clusters':2}
    assert result['n_missing']==1 and model['coverage']['missing_outputs']==1
    paired=next(r for r in result['paired_comparisons'] if r['metric']=='ai_prompt_adherence')
    assert paired['n_clusters']==2 and paired['matched_outputs']==2
    file.write_text(file.read_text()+json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='duplicate IDs'): aggregate(dest)


def test_reconstruction_cannot_invent_unseen_surface_correctness_or_advertising_approval():
    from stylesphere.reporting.ai_visual import RECON_SCORES
    card={'assessment_id':'synthetic-recon','pair':'reconstruction','criteria':list(RECON_CRITERIA),
          'display_assets':{'view_00':{}},'evidence_sha256':{'view_00':'1'*64}}
    row=assessment_template(card,'protocol')
    row.update(assessor_id='synthetic-unit-reviewer',assessor_model='not_exposed',assessed_at_utc='2026-01-01T00:00:00Z',
               review_status='assessed',confidence='low',scores={name:3 for name in RECON_SCORES},
               criteria=[{'criterion':name,'verdict':'uncertain','evidence':'No reference geometry'} for name in RECON_CRITERIA],
               observations=[{'asset':'view_00','description':'Synthetic unit fixture only'}])
    validate_assessment(row,card,'protocol')
    row['criteria'][-1]['verdict']='met'
    with pytest.raises(ValueError,match='Unseen surface'): validate_assessment(row,card,'protocol')
    row['criteria'][-1]['verdict']='uncertain';row['ai_usable_concept']=1
    with pytest.raises(ValueError,match='Reconstruction cannot'): validate_assessment(row,card,'protocol')
