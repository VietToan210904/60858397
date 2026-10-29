"""Regenerate the final four-pair synthesis from existing retained evidence."""
from pathlib import Path
import json
from stylesphere.reporting.final_report import build

if __name__=='__main__':
    print(json.dumps(build(Path.cwd())))
