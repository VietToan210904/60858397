"""Synthetic package fixtures exercise exclusion and integrity controls only."""
import json
import hashlib
import zipfile
import pytest
from stylesphere.reporting import packaging


def test_licence_asset_exclusion_and_hash_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(packaging, 'verify', lambda *args: {'integrity_passed':True, 'pending':['Labelled test fixture']})
    data=tmp_path/'datasets'/'retrieval'
    data.mkdir(parents=True)
    (data/'private.png').write_bytes(b'synthetic-not-an-image')
    (data/'public.png').write_bytes(b'synthetic-not-an-image')
    (data/'catalogue.csv').write_text('product_id,image_path,redistributable\np1,datasets/retrieval/private.png,false\np2,datasets/retrieval/public.png,true\n')
    (tmp_path/'README.md').write_text('Synthetic unit-test package only')
    output=tmp_path/'results'
    output.mkdir()
    result=packaging.package(tmp_path,output)
    with zipfile.ZipFile(result['zip']) as z:
        assert 'StyleSphere_Evaluation/datasets/retrieval/private.png' not in z.namelist()
        assert 'StyleSphere_Evaluation/datasets/retrieval/public.png' in z.namelist()
        manifest=json.loads(z.read('StyleSphere_Evaluation/SUBMISSION_MANIFEST.json'))
        assert manifest['label']=='PARTIAL_RESEARCH_PACKAGE'
        assert manifest['omitted_assets'][0]['path'].endswith('private.png')
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(z.read('StyleSphere_Evaluation/' + name)).hexdigest() == expected


def test_integrity_errors_stop_packaging(tmp_path, monkeypatch):
    monkeypatch.setattr(packaging, 'verify', lambda *args: {'integrity_passed':False})
    with pytest.raises(ValueError,match='integrity'):
        packaging.package(tmp_path,tmp_path/'results')


def test_credential_in_csv_stops_packaging(tmp_path,monkeypatch):
    monkeypatch.setattr(packaging, 'verify', lambda *args: {'integrity_passed':True,'pending':[]})
    output=tmp_path/'results'
    output.mkdir()
    (output/'records.csv').write_text('value\n'+'hf_'+'a'*25)
    with pytest.raises(ValueError,match='credential'):
        packaging.package(tmp_path,output)


def test_completed_component_requires_valid_receipt_and_preserves_legacy_scope(tmp_path, monkeypatch):
    from stylesphere.reporting import completion
    from stylesphere.reporting import final_report
    monkeypatch.setattr(packaging, 'verify', lambda *args: {'integrity_passed':True,'pending':['Independent human assessment absent']})
    monkeypatch.setattr(completion, 'validate_completion_receipt', lambda root: [])
    monkeypatch.setattr(final_report, 'validate_report_snapshot', lambda root: [])
    output=tmp_path/'results'
    final=output/'final_evaluation'
    final.mkdir(parents=True)
    (final/'completion.json').write_text('{"synthetic_test_receipt":true}')
    result=packaging.package(tmp_path,output)
    assert result['requested_component_complete'] is True
    assert result['zip'].endswith('StyleSphere_Evaluation.zip')
    with zipfile.ZipFile(result['zip']) as archive:
        manifest=json.loads(archive.read('StyleSphere_Evaluation/SUBMISSION_MANIFEST.json'))
        assert manifest['label']=='COMPLETE_REQUESTED_COMPONENT'
        assert manifest['benchmark_complete'] is False
        assert manifest['pending']==[]
        assert manifest['legacy_verifier_findings']==['Independent human assessment absent']


def test_stale_completion_receipt_stops_final_archive(tmp_path, monkeypatch):
    from stylesphere.reporting import completion
    monkeypatch.setattr(packaging, 'verify', lambda *args: {'integrity_passed':True,'pending':[]})
    monkeypatch.setattr(completion, 'validate_completion_receipt', lambda root: ['Evidence changed'])
    output=tmp_path/'results'
    (output/'final_evaluation').mkdir(parents=True)
    (output/'final_evaluation/completion.json').write_text('{}')
    with pytest.raises(ValueError,match='completion evidence is invalid'):
        packaging.package(tmp_path,output)
    assert not (tmp_path/'submission/StyleSphere_Evaluation.zip').exists()
