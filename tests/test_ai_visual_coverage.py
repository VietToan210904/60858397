"""Synthetic decisions verify unresolved denominators without score imputation."""
from stylesphere.reporting.ai_visual_coverage import decision_coverage


def test_uncertain_decisions_remain_visible_and_complete_prompt_subset_is_paired():
    mapping, rows = [], []
    for model, values in [('a',[1,1,None,0]),('b',[0,1,1,0])]:
        for index, value in enumerate(values):
            identity=f'{model}-{index}'
            mapping.append({'assessment_id':identity,'pair':'generation','model':model,
                            'case_id':'p1' if index<3 else 'p2','seed':index,'input_sha256':str(index)})
            rows.append({'assessment_id':identity,'review_status':'assessed','ai_usable_concept':value})
    result=decision_coverage(mapping,rows)
    a=result['by_model']['a']
    assert (a['planned'],a['positive'],a['negative'],a['uncertain'],a['decided'])==(4,2,1,1,3)
    assert a['possible_all_planned_rate_bounds']==[.5,.75]
    assert result['matched_pair_coverage']['complete_decision_pairs']==3
    assert result['complete_prompt_subset']=={'n_complete_prompts':1,'n_input_seed_pairs':1,
                                               'ai_positive_prompt_mean':{'a':0.,'b':0.}}
