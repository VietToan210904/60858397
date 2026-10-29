"""Hash-check and extract a submission, then run its copied-source CPU validators."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--receipt',type=Path,default=Path('submission/extraction-validation.json'))
    args=parser.parse_args()
    task_root=Path.cwd().resolve()
    destination=(args.destination or Path('.cache/package-validation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')).resolve()
    if not destination.is_relative_to(task_root) or destination.exists():
        raise ValueError('Use a fresh extraction directory inside this workspace; no existing directory is replaced.')
    destination.mkdir(parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        infos=archive.infolist()
        names=[x.filename for x in infos]
        if len(names)!=len(set(names)):
            raise ValueError('Duplicate archive member')
        for entry in infos:
            name=PurePosixPath(entry.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in entry.filename or name.parts[0]!='StyleSphere_Evaluation' or (entry.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe archive member')
        size=sum(x.file_size for x in infos)
        if size>20_000_000_000 or shutil.disk_usage(destination).free<size*2:
            raise ValueError('Insufficient disk space or unreasonable expanded size')
        manifest=json.loads(archive.read('StyleSphere_Evaluation/SUBMISSION_MANIFEST.json'))
        expected={'StyleSphere_Evaluation/'+name for name in manifest['files']}
        if set(names)!=expected|{'StyleSphere_Evaluation/SUBMISSION_MANIFEST.json'}:
            raise ValueError('Archive membership differs from its exact hash manifest')
        archive.extractall(destination)
    copied_root=destination/'StyleSphere_Evaluation'
    for name,expected in manifest['files'].items():
        if digest(copied_root/name)!=expected:
            raise ValueError('Extracted hash differs: '+name)
    child_result=destination/'copied-source-validation.json'
    child=r'''
import json,sys
from pathlib import Path
root=Path.cwd().resolve()
import stylesphere.reporting.verification as verification
from stylesphere.reporting.completion import validate_completion_receipt
from stylesphere.reporting.final_report import validate_report_snapshot
if not Path(verification.__file__).resolve().is_relative_to(root/'src'):
    raise RuntimeError('Validation imported original rather than copied source')
legacy=verification.verify(root,root/'results')
errors=list(legacy['errors'])+validate_completion_receipt(root)+validate_report_snapshot(root)
result={'copied_source':str(Path(verification.__file__).resolve()),'integrity_passed':not errors,
        'checked_original_records':legacy['checked_records'],'errors':errors,
        'requested_component_complete':not errors,'legacy_primary_benchmark_complete':False,
        'legacy_validation_findings':len(legacy['pending'])}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
raise SystemExit(0 if not errors else 1)
'''
    environment=dict(os.environ,PYTHONPATH=str(copied_root/'src'),PYTHONUTF8='1')
    run=subprocess.run([sys.executable,'-c',child,str(child_result)],cwd=copied_root,env=environment,capture_output=True,text=True)
    if not child_result.exists():
        raise RuntimeError('Copied-source validation failed: '+run.stderr[-5000:])
    result=json.loads(child_result.read_text(encoding='utf-8'))
    result.update(archive_sha256=digest(args.archive.resolve()),archive_bytes=args.archive.stat().st_size,
                  extracted_root=str(copied_root),archive_member_count=len(names),
                  member_hashes_verified_before_validation=len(manifest['files']),expanded_bytes=size,
                  subprocess_returncode=run.returncode,validated_at_utc=datetime.now(timezone.utc).isoformat(),
                  note='All archived file hashes matched after fresh extraction. Copied-source verification then regenerates only its own verification receipt; original project evidence is untouched.')
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('integrity_passed','requested_component_complete','archive_member_count','member_hashes_verified_before_validation','checked_original_records','errors')}))
    if run.returncode:
        raise SystemExit(run.returncode)


if __name__=='__main__':
    main()
