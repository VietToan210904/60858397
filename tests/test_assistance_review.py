"""Synthetic human-review schema tests in isolated temporary directories."""
import csv
import pytest
from stylesphere.assistance.review import FIELDS, review_assistance


def test_review_blanks_and_actual_numeric_ratings(tmp_path):
    record={'run_id':'synthetic-fixture','pair':'assistance','model':'a','case_id':'case1','seed':1,'status':'success','mode':'evaluation','protocol_sha256':'test','input_sha256':'test'}
    path=tmp_path/'review.csv'
    first=review_assistance([record],path)
    assert first['n_assessments']==0 and first['models']==[]
    with path.open(newline='') as stream: rows=list(csv.DictReader(stream))
    assert rows[0]['task_success']==''
    rows[0].update(rater_id='synthetic-test-rater',task_success='0',factual_prose_correct='1')
    def save():
        with path.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    save()
    result=review_assistance([record],path)
    assert result['models'][0]['metrics']['task_success']['mean']==0
    rows[0]['record_sha256']='tampered'; save()
    with pytest.raises(ValueError,match='retained execution'): review_assistance([record],path)


def test_disagreeing_human_votes_are_not_half_a_completed_task(tmp_path):
    record={'run_id':'synthetic-fixture','pair':'assistance','model':'a','case_id':'case1','seed':1,'status':'success','mode':'evaluation','protocol_sha256':'test','input_sha256':'test'}
    path=tmp_path/'review.csv'
    review_assistance([record],path)
    with path.open(newline='') as stream:
        template=next(csv.DictReader(stream))
    rows=[{**template,'rater_id':'test-rater-a','task_success':'0'},
          {**template,'rater_id':'test-rater-b','task_success':'1'}]
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    result=review_assistance([record],path)
    case=result['per_case'][0]
    assert 'task_success' not in case['metrics']
    assert case['supplemental_vote_rates']['task_success']==.5
    assert case['disagreement_fields']==['task_success']
