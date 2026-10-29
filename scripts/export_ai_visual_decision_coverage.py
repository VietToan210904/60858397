"""Validate existing AI annotations and expose all shortlist decision denominators."""
import json
from pathlib import Path

from stylesphere.common.io import read_jsonl, sha256_file, utc_now, write_json
from stylesphere.reporting.ai_visual import aggregate
from stylesphere.reporting.ai_visual_coverage import decision_coverage


root = Path(__file__).resolve().parents[1]
destination = root/'results/ai_visual_assessment'
validated = aggregate(destination)
mapping = json.loads((destination/'private_mapping.json').read_text())
rows = [row for path in sorted((destination/'assessments').glob('*.jsonl')) for row in read_jsonl(path)]
result = decision_coverage(mapping, rows)
result.update(generated_at_utc=utc_now(), protocol_sha256=validated['protocol_sha256'],
              selection_sha256=validated['selection_sha256'], assessment_files_sha256=validated['assessment_files_sha256'],
              helper_source_sha256=sha256_file(root/'src/stylesphere/reporting/ai_visual_coverage.py'))
write_json(destination/'decision_coverage.json',result)
print(json.dumps({key:result[key] for key in ('by_model','matched_pair_coverage','complete_prompt_subset')},indent=2))
