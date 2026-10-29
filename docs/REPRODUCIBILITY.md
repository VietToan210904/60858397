# Reproducibility checklist

1. Read AI_ASSISTANCE_DISCLOSURE.md, ASSIGNMENT_ALIGNMENT.md and
   IMPLEMENTATION_STATUS.md. The user confirmed the four-pair/eight-model
   Evaluation Codes and Data scope and permitted AI assistance. The original
   584 attempts and separate AI/reference analyses are recorded. Completion of
   this component does not imply independent human validation; AI labels remain
   explicitly AI labels.
2. From the project root, use Python 3.11 and an isolated environment. Run
   `scripts/setup_windows.ps1 -Pair retrieval -InstallCuda -ApproveLargeDownload`
   only after approving the download budget. This modifies no drivers/toolkits.
3. Run `python -m stylesphere.cli doctor`. Confirm CUDA in **this interpreter**.
   Retain initial and later environment reports separately.
4. Run `python -m pytest -q`. Unit-test examples are synthetic and never enter
   result records. Save the output and exact package versions.
5. Check pinned identities in config/models.yaml and official sources in docs.
   `acquire --model fashion_siglip --dry-run` reports exact sizes and disk space.
   Run `acquire --model fashion_siglip`, `acquire --model siglip2`, and
   `acquire --model fashion_tokenizer` after approval. Retain acquisition SHA256s.
6. For a new smoke set with no manifest, run
   `python scripts/acquire_retrieval_smoke.py`. To restore images omitted from the ZIP
   while preserving its pinned manifest, run
   `python scripts/acquire_retrieval_smoke.py --restore-missing` from the project root.
   In PowerShell with the retrieval interpreter, use
   `& $python scripts/acquire_retrieval_smoke.py --restore-missing`.
   Existing valid images and all input metadata are preserved. Restoration requires
   the recorded revision and exact SHA256s and stops above 100 MB, on changed content,
   or if that pinned source is unavailable. It never substitutes a new revision.
   Default mode strictly hash-checks existing images and errors if any are missing.
   Dataset asset redistribution is not assumed; the ZIP omits unapproved images.
   Actual isolated restoration verified 8 images and 162,603 transferred response bytes,
   with all hashes matched and the source/copied manifests unchanged; see
   `environment/smoke_restore_validation.json`.
7. Run `python -m stylesphere.cli smoke --pair retrieval --config config/retrieval_smoke.yaml`.
   No relevance scores are available for these unjudged images. Smoke output never
   counts as the final benchmark.
8. For a new experiment, acquire real assets under their terms and freeze inputs,
   deployment settings and the scoring protocol before final cases. Existing
   `prepare --all` preserves populated manifests; it does not regenerate frozen
   inputs. Independent human judgements belong in separate actual-review files.
   The completed AI supplements and GSO extension have their own frozen versions;
   see the recomputation commands below. Never tune on final scores or overwrite
   original raw records to make a later analysis appear prespecified.
9. Set up each other environment separately. Check model gates and download sizes
   before acquiring assets. Configure reviewed checkout commits and paths for 3D
   and FLUX FP8. Execute pair smoke tests before full evaluation.
   For rendered reconstruction previews, use the `.venv-preview` parent CLI and
   isolated SF/SP model workers described in `NATIVE_3D_SETUP.md`.
10. Run `evaluate --pair retrieval` and analogous pair commands, or use
    `scripts/run_all_evaluations.ps1` to dispatch isolated interpreters sequentially.
    Preserve failures and resume only matching cases with valid artifacts.
11. Preserve actual AI decisions and their separate provenance. If collecting new
    independent human reviews, use the still-blank human worksheets, keep model
    mappings from raters until ratings are frozen and document adjudication.
12. Rebuild reports and run both integrity/completion verification before packaging.
    The legacy verifier's independent-human requirements remain distinct from
    completion of the user-confirmed component. Inspect the exact report,
    completion receipt and ZIP manifest rather than treating any single exit code
    as evidence of scientific validity.

The execution log in environment/ records which commands actually ran. All eight
models' planned computational cases were attempted. Additional completed analyses
cover 600 retrieval AI profiles, 184 AI visual assessments, 160 AI shopping reviews
and 40 GSO test plus eight development model outputs. Human assessments remain
blank and the original ABO reference fields remain unavailable. Actual
native/preview installs and GPU runs were
tested on this workstation, while separate cold rebuilds of the source/dependency
versions were not validated. The final 3D source/input/deployment freeze is in
`environment/reconstruction-deployment-freeze.json`; its resume receipt confirms
all 40 cases were reused with no new inference or raw-record change. Exact package
origins/versions and build receipts are retained in environment/. Version ranges
express compatibility expectations rather than universal locks.

## Recompute the separate supplements

Run from the extracted project root using a compatible project environment. The
`.venv-retrieval` and `.venv-preview` paths below identify the tested workstation
interpreters; virtual environments and weights are excluded from the ZIP. A fresh
environment must be installed from the documented dependencies before these
commands work. These commands reuse retained decisions and model outputs; they
do not create a new independent assessment or rerun model inference.

```powershell
$reviewPython = '.\.venv-retrieval\Scripts\python.exe'
& $reviewPython -m stylesphere.retrieval.proxy_metrics qrels
& $reviewPython -m stylesphere.retrieval.proxy_metrics score --run-id 24512250744c475fb358342058a8a043
& $reviewPython scripts/export_ai_visual_review.py validate
& $reviewPython scripts/export_ai_visual_review.py summarize
& $reviewPython scripts/export_ai_visual_decision_coverage.py
& $reviewPython -m stylesphere.assistance.ai_assessment summarize
& .\.venv-preview\Scripts\python.exe -m stylesphere.reconstruction.reference_benchmark verify
& $reviewPython -m stylesphere.reporting.final_report
& $reviewPython -m stylesphere.reporting.completion inspect
```

Recomputing summaries can update generated timestamps/manifests. Rebuild the final
report and completion receipt after the last intended change, then package those
exact bytes. Do not change an authored ledger, profile freeze or existing rating
merely to make verification pass. `materialize` in the assistance supplement can
rebuild its expanded assessment JSONL from the retained explicit authored ledger;
it is serialization of existing decisions, not automatic prose judgement.

The GSO extension retains four disjoint development objects and 20 test objects
per model, raw mesh outputs, separate surface samples, transforms and saved
reference cameras. To regenerate its scores from the existing meshes:

```powershell
& .\.venv-preview\Scripts\python.exe -m stylesphere.reconstruction.reference_benchmark score --split development
& .\.venv-preview\Scripts\python.exe -m stylesphere.reconstruction.reference_benchmark score --split test
& .\.venv-preview\Scripts\python.exe -m stylesphere.reconstruction.reference_benchmark verify
```

Scoring includes local OpenGL silhouette rendering and CPU geometry calculations;
coordinate any active GPU benchmark before doing it. No checkpoint inference is
performed by `score`. The original normalization and reference-assisted ICP
settings are frozen and must not be adjusted after viewing test outcomes. Full
acquisition/preparation and sequential inference commands are in the separate
[reference reconstruction protocol](REFERENCE_RECONSTRUCTION.md).

After report/source bytes are final:

```powershell
& $reviewPython -m stylesphere.reporting.completion finalize
& $reviewPython -m stylesphere.reporting.completion verify
& $reviewPython -m stylesphere.cli verify --config config/local.yaml
& $reviewPython -m stylesphere.cli package --config config/local.yaml
```

Inspect each result and independently verify the extracted archive. Completion
means the declared component's required evidence is present and hash-bound;
it does not establish human preference, absolute physical geometry accuracy or
commercial approval.

Historical raw records keep their original paths and bytes. The audited
`environment/path_origins.json` map lets verification resolve an original absolute
workstation artifact path to the corresponding file under a freshly extracted
project root. This does not rewrite predictions or alter their hashes. New worker
outputs use relative POSIX paths directly. Validate the extracted ZIP independently
before treating it as a reproducible snapshot.
