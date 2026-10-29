"""Decision-denominator diagnostics, separate from the frozen AI packet exporter."""
from collections import Counter, defaultdict


def decision_coverage(mapping, assessments):
    """Expose unresolved AI shortlist opinions without imputing negative decisions."""
    lookup = {}
    for row in assessments:
        if row['assessment_id'] in lookup:
            raise ValueError('Duplicate primary AI assessment')
        lookup[row['assessment_id']] = row
    planned = [row for row in mapping if row['pair'] == 'generation']
    models, prompts, paired = defaultdict(Counter), defaultdict(Counter), defaultdict(dict)
    for info in planned:
        row = lookup.get(info['assessment_id'])
        value = row.get('ai_usable_concept') if row else None
        state = ('missing' if row is None else 'abstained' if row['review_status'] == 'abstained'
                 else 'uncertain' if value is None else 'positive' if value == 1 else 'negative')
        for count in (models[info['model']], prompts[(info['model'], info['case_id'])]):
            count['planned'] += 1
            count[state] += 1
        paired[(info['case_id'], info['seed'], info['input_sha256'])][info['model']] = value
    def counts(count):
        result = {key: count.get(key, 0) for key in ('planned','positive','negative','uncertain','abstained','missing')}
        decided = result['positive'] + result['negative']
        unresolved = result['planned'] - decided
        return {**result,'decided':decided,'unresolved':unresolved,
                'conditional_image_positive_rate': result['positive']/decided if decided else None,
                'possible_all_planned_rate_bounds': [result['positive']/result['planned'],
                                                    (result['positive']+unresolved)/result['planned']]}
    names = sorted(models)
    pairs, prompt_pairs = Counter(), defaultdict(list)
    if len(names) == 2:
        for (case,seed,input_sha), values in paired.items():
            complete = all(name in values and values[name] is not None for name in names)
            pairs['planned_matched_input_seed_pairs'] += 1
            pairs['complete_decision_pairs' if complete else 'unresolved_decision_pairs'] += 1
            if complete:
                pairs[f'{names[0]}_{int(values[names[0]])}__{names[1]}_{int(values[names[1]])}'] += 1
            prompt_pairs[case].append((complete, values))
    complete_prompts = {case:items for case,items in prompt_pairs.items() if all(item[0] for item in items)}
    complete_subset = {'n_complete_prompts':len(complete_prompts),
                       'n_input_seed_pairs':sum(len(items) for items in complete_prompts.values()),
                       'ai_positive_prompt_mean':{name:sum(sum(v[name] for _,v in items)/len(items)
                                                          for items in complete_prompts.values())/len(complete_prompts)
                                                  if complete_prompts else None for name in names}}
    return {'kind':'post_hoc_ai_shortlist_denominator_diagnostics',
            'by_model':{name:counts(value) for name,value in sorted(models.items())},
            'per_prompt':[{'model':model,'case_id':case,**counts(value)} for (model,case),value in sorted(prompts.items())],
            'matched_pair_coverage':dict(pairs),'complete_prompt_subset':complete_subset,
            'note':'Null decisions remain unresolved. Conditional rates have different observed denominators; bounds are explicit worst/best unresolved-decision scenarios, not imputed scores. Complete-prompt comparisons are an additional coverage diagnostic and do not replace the frozen summary or human assessment.'}
