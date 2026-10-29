"""A stale synthesis cannot be packaged as the final evaluated component."""
import hashlib
import json
from stylesphere.reporting.final_report import validate_report_snapshot


def test_changed_source_or_displayed_report_invalidates_final_snapshot(tmp_path):
    folder=tmp_path/'results/final_evaluation'
    folder.mkdir(parents=True)
    source=tmp_path/'source.json'
    source.write_text('{"synthetic_test":true}')
    generated={}
    for name in ('REPORT.md','summary.json','deployment_results.csv','quality_overview.png'):
        data=b'synthetic-unit-test-only'
        (folder/name).write_bytes(data)
        generated[name]=hashlib.sha256(data).hexdigest()
    manifest={'source_sha256':{'source.json':hashlib.sha256(source.read_bytes()).hexdigest()},'generated_sha256':generated}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    assert validate_report_snapshot(tmp_path)==[]
    source.write_text('{"changed":true}')
    assert any('source.json' in error for error in validate_report_snapshot(tmp_path))
    manifest['source_sha256']['source.json']=hashlib.sha256(source.read_bytes()).hexdigest()
    (folder/'manifest.json').write_text(json.dumps(manifest))
    (folder/'REPORT.md').write_text('stale or manually edited conclusion')
    assert any('REPORT.md' in error for error in validate_report_snapshot(tmp_path))
