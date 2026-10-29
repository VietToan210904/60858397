"""Synthetic unit-test fixtures; never benchmark observations."""
import copy
import json
from pathlib import Path
import pytest
from stylesphere.common.config import load_config, validate_config
from stylesphere.common.io import append_jsonl, portable_path, read_jsonl, scrub, sha256_file, write_json
from stylesphere.common.runtime import Measurement
from stylesphere.common.statistics import paired_bootstrap, summary
from stylesphere.reporting.reports import latest_records, report

ROOT=Path(__file__).resolve().parents[1]


def test_configuration_override(tmp_path):
    config=load_config(ROOT)
    validate_config(config)
    invalid=copy.deepcopy(config)
    invalid['models']['siglip2']['model_id']=invalid['models']['fashion_siglip']['model_id']
    with pytest.raises(ValueError): validate_config(invalid)
    path=tmp_path/'override.yaml'
    path.write_text('seed: 42\n',encoding='utf-8')
    assert load_config(ROOT,path)['seed']==42


def test_io_and_secret_redaction(tmp_path):
    path=tmp_path/'nested'/'values.jsonl'
    append_jsonl(path,{'api_key':'private','tokenizer_revision':'abc','generated_tokens':3})
    assert read_jsonl(path)==[{'api_key':'[REDACTED]','tokenizer_revision':'abc','generated_tokens':3}]
    write_json(tmp_path/'atomic.json',{'a':1})
    assert len(sha256_file(path))==64
    path.write_text('{bad}\n',encoding='utf-8')
    with pytest.raises(ValueError): read_jsonl(path)


def test_cpu_timer_failure_still_measured():
    timer=Measurement(cuda=False)
    with pytest.raises(RuntimeError):
        with timer: raise RuntimeError('labelled synthetic failure')
    assert timer.elapsed_s>=0 and timer.peak_allocated_mb is None


def test_historical_windows_artifact_resolves_after_extraction(tmp_path):
    write_json(tmp_path / 'environment/path_origins.json', {'original_project_roots':['C:/Users/example/original-project']})
    assert portable_path(tmp_path, r'C:\Users\example\original-project\results\image.png') == tmp_path / 'results/image.png'
    assert portable_path(tmp_path, r'results\image.png') == tmp_path / 'results/image.png'


def test_bootstrap_hand_calculated_and_clustered():
    result=paired_bootstrap([3,5],[1,3],iterations=100,seed=1)
    assert result['mean_difference_a_minus_b']==2
    assert result['ci_low']==result['ci_high']==2
    clustered=paired_bootstrap([2,2,10],[0,0,0],clusters=['p1','p1','p2'],iterations=100)
    assert clustered['mean_difference_a_minus_b']==6 and clustered['n_clusters']==2
    relative=paired_bootstrap([2,2,10],[1,1,2],clusters=['p1','p1','p2'],iterations=100)
    # Equal prompt weights: means 6 and1.5, difference4.5, relative change3.
    assert relative['mean_difference_a_minus_b']==4.5 and relative['relative_difference']==3
    assert paired_bootstrap([1],[0])['ci_low'] is None
    with pytest.raises(ValueError): paired_bootstrap([1],[2,3])
    assert summary([])['mean'] is None


def test_success_preserved_and_config_separated():
    base={'pair':'retrieval','model':'a','case_id':'q1','seed':1,'mode':'evaluation','config_sha256':'a','input_sha256':'b'}
    values=latest_records([{**base,'status':'success'},{**base,'status':'failed'},{**base,'status':'success','config_sha256':'c'}])
    assert len(values)==2 and all(r['status']=='success' for r in values)


def test_empty_report_truthful(tmp_path):
    output=tmp_path/'results'
    result=report(tmp_path,output,{'seed':8410})
    assert result['records']==0
    assert 'No successful benchmark' in (output/'reports'/'REPORT.md').read_text()
    assert not list((output/'reports').glob('*.png'))


def test_paired_reports_match_protocol_and_cluster(tmp_path):
    output=tmp_path/'results'
    for model,score in [('a',1),('b',0)]:
        for seed in [1,2,3]:
            append_jsonl(output/'generation'/'records.jsonl', {'pair':'generation','model':model,'case_id':'prompt1','seed':seed,'status':'success','mode':'evaluation','provenance':'model_inference','protocol_sha256':'same','config_sha256':model,'input_sha256':'input','metrics':{'valid_output':score},'timings':{}})
    result=report(tmp_path,output,{'seed':1,'evaluation':{'statistics':{'bootstrap_iterations':10}}})
    comparison=json.loads((output/'reports'/'paired_comparisons.json').read_text())
    valid_output = next(row for row in comparison if row['metric'] == 'valid_output')
    assert valid_output['n_clusters']==1 and valid_output['matched_cases']==3


def test_reliability_comparison_keeps_real_execution_failures(tmp_path):
    output = tmp_path / 'results'
    for model, status in [('a', 'success'), ('b', 'failed')]:
        append_jsonl(output / 'assistance/records.jsonl', {'pair':'assistance','model':model,'case_id':'case1',
                     'seed':1,'mode':'evaluation','status':status,'provenance':'model_inference',
                     'protocol_sha256':'same','input_sha256':'same','config_sha256':model,'run_id':'synthetic-run'})
    report(tmp_path, output, {'seed':8410})
    comparisons = json.loads((output / 'reports/paired_comparisons.json').read_text())
    reliability = next(row for row in comparisons if row['metric'] == 'execution_success_rate')
    assert reliability['mean_difference_a_minus_b'] == 1
    assert reliability['matched_cases'] == 1 and reliability['ci_low'] is None


def test_recovery_does_not_erase_attempt_failure_or_first_attempt_comparison(tmp_path):
    import csv
    output = tmp_path / 'results'
    common = {'pair':'generation','case_id':'case1','seed':1,'mode':'evaluation',
              'provenance':'model_inference','protocol_sha256':'same','input_sha256':'same'}
    for model, status in [('a','success'),('b','failed'),('b','success')]:
        append_jsonl(output / 'generation/records.jsonl', {**common,'model':model,'status':status,'config_sha256':model})
    report(tmp_path, output, {'seed':8410})
    with (output / 'reports/model_summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    failure = next(row for row in rows if row['model']=='b' and row['metric']=='execution_failure_rate')
    assert int(failure['n'])==2 and float(failure['mean'])==.5
    comparisons = json.loads((output / 'reports/paired_comparisons.json').read_text())
    reliability = next(row for row in comparisons if row['metric']=='execution_success_rate')
    assert reliability['mean_difference_a_minus_b']==1
    with (output / 'reports/failures.csv').open() as stream:
        assert len(list(csv.DictReader(stream)))==1


def test_dataset_hash_changes_fail_integrity_before_inference(tmp_path, monkeypatch):
    from stylesphere.reporting.verification import verify
    import stylesphere.datasets
    monkeypatch.setattr(stylesphere.datasets, 'validate', lambda root, pair:
                        ['Image hash differs: changed-product', 'No human judgements for query-1',
                         'Unknown/duplicate qrels ids: invalid-fixture'] if pair == 'retrieval' else [])
    result = verify(tmp_path, tmp_path / 'results')
    assert 'Dataset retrieval: Image hash differs: changed-product' in result['errors']
    assert 'Dataset retrieval: No human judgements for query-1' in result['pending']
    assert 'Dataset retrieval: Unknown/duplicate qrels ids: invalid-fixture' in result['errors']


def test_retained_failure_artifact_tampering_is_an_integrity_error(tmp_path):
    from stylesphere.reporting.verification import verify
    artifact = tmp_path / 'results/retrieval/failure.json'
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"failure":"synthetic test diagnostic"}')
    expected = sha256_file(artifact)
    append_jsonl(tmp_path / 'results/retrieval/records.jsonl', {
        'pair':'retrieval','model':'synthetic-test','case_id':'test-case','seed':1,
        'status':'failed','mode':'evaluation','provenance':'model_inference',
        'run_id':'test-run','timestamp_utc':'2026-01-01T00:00:00Z',
        'checkpoint_revision':'a'*40,'input_sha256':'input','config_sha256':'config',
        'artifacts':{'diagnostic':'results/retrieval/failure.json'},
        'artifact_sha256':{'diagnostic':expected}})
    artifact.write_text('{"failure":"changed"}')
    result = verify(tmp_path, tmp_path / 'results')
    assert any('Changed artifact:' in error for error in result['errors'])


def test_generation_throughput_includes_measured_failed_attempts(tmp_path):
    output = tmp_path / 'results'
    common = {'pair':'generation','model':'a','seed':1,'mode':'evaluation',
              'provenance':'model_inference','config_sha256':'config','input_sha256':'input'}
    append_jsonl(output / 'generation/records.jsonl', {**common,'case_id':'ok','status':'success',
                 'metrics':{'valid_output':1},'timings':{'inference_s':2}})
    append_jsonl(output / 'generation/records.jsonl', {**common,'case_id':'failure','status':'failed',
                 'metrics':{},'timings':{'inference_s':3}})
    report(tmp_path, output, {'seed':8410})
    import csv
    rows = list(csv.DictReader((output / 'reports/model_summary.csv').open()))
    throughput = next(r for r in rows if r['metric'] == 'valid_outputs_per_inference_second')
    assert int(throughput['n']) == 2 and float(throughput['mean']) == pytest.approx(.2)


def test_generation_throughput_unavailable_if_failure_has_no_timing(tmp_path):
    import csv
    output=tmp_path/'results'
    common={'pair':'generation','model':'a','seed':1,'mode':'evaluation','provenance':'model_inference','config_sha256':'a','input_sha256':'x'}
    append_jsonl(output/'generation/records.jsonl',{**common,'case_id':'ok','status':'success','metrics':{'valid_output':1},'timings':{'inference_s':2}})
    append_jsonl(output/'generation/records.jsonl',{**common,'case_id':'failed','status':'failed','metrics':{},'timings':{}})
    report(tmp_path,output,{'seed':8410})
    with (output/'reports/model_summary.csv').open() as stream:
        row=next(row for row in csv.DictReader(stream) if row['metric']=='valid_outputs_per_inference_second')
    assert row['mean']=='' and int(row['untimed_attempts'])==1


def test_paired_analysis_rejects_ambiguous_deployment_configurations(tmp_path):
    output=tmp_path/'results'
    for configuration in ('first','changed'):
        append_jsonl(output/'generation/records.jsonl',{'pair':'generation','model':'a','case_id':'case1','seed':1,
                     'mode':'evaluation','provenance':'model_inference','status':'success',
                     'config_sha256':configuration,'protocol_sha256':'same','input_sha256':'same'})
    with pytest.raises(ValueError,match='deployment configurations'):
        report(tmp_path,output,{'seed':8410})
