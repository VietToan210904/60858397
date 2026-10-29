# StyleSphere Evaluation

A local, auditable Python framework for four paired comparisons in a hypothetical
fashion retailer. It evaluates models, rather than implementing a shopping website.
The repository contains real inference adapters, controlled inputs, mathematical
metrics, failure records, review workflows and reproducible reporting.

The confirmed **four-pair/eight-model Evaluation Codes and Data component is complete**.
Start with the [final results and recommendations](results/final_evaluation/REPORT.md),
[validated completion receipt](results/final_evaluation/completion.json) and
[implementation status](docs/IMPLEMENTATION_STATUS.md). All eight exact models ran
locally on the RTX 4070 SUPER. Quality assessments are explicitly labelled AI
opinions/proxies; a separate 3D extension uses genuine licensed scanned references.
Independent human worksheets remain blank and are not claimed as completed feedback.

All 584 planned technical cases have been attempted across the eight selected
models: 240 retrieval queries, 144 campaign images, 160 shopping attempts and
40 reconstruction cases. There were 511 technical completions and 73 strict
shopping-protocol failures (Phi 40, Qwen 33). Both generation models produced 72
images each. Both 3D models exported 20/20 GLBs with four RGB/mask preview pairs
per object and no execution failures. Strict mesh validity was
19/20 for SF3D and 20/20 for SPAR3D; the SF exception remains in the evidence.
Both gated checkpoints were officially acquired and SHA-verified. All 40 3D cases
subsequently resumed without new inference or a raw-record hash change.
The completed supplements add AI review of all 600 retrieval photos, 144 generated
images, 160 shopping traces and 40 original 3D preview sets. The new controlled
reference extension adds 40 test reconstructions and 8 development reconstructions,
with normalized geometry and explicitly reference-assisted rigid alignment scores.
The original ABO photos still lack aligned reference geometry. See the
[native execution timeline and timing qualifications](docs/NATIVE_3D_SETUP.md#executed-development-and-final-reconstruction).

**Confirmed scope and disclosure:** the user explicitly confirmed four pairs /
eight models and stated AI use is permitted and encouraged. That request governs
this component; different wording in the original PDF is historical context.
See [scope clarification](docs/ASSIGNMENT_ALIGNMENT.md) and
[AI disclosure](docs/AI_ASSISTANCE_DISCLOSURE.md).

## Models and deployment configurations

| Pair / task | System A | System B | Important asymmetry |
|---|---|---|---|
| 1: text-to-product retrieval | `Marqo/marqo-fashionSigLIP` | `google/siglip2-base-patch16-224` | Fashion adaptation versus general encoder; model-native preprocessing |
| 2: campaign image concepts | `ByteDance/SDXL-Lightning`, exact 4-step UNet | `black-forest-labs/FLUX.2-klein-4b-fp8` | Euler trailing / guidance 0 versus exact scaled FP8 distilled 4B deployment |
| 3: grounded shopping assistance | `microsoft/Phi-4-mini-instruct` | `Qwen/Qwen3.5-4B` | Same text-only tasks/tools; Qwen thinking disabled, unused vision tower retained |
| 4: single-image 3D | `stabilityai/stable-point-aware-3d` | `stabilityai/stable-fast-3d` | SPAR point diffusion / low-VRAM mode; separate geometry and texture dependencies |

Full 40-character checkpoint revisions are pinned in [models.yaml](config/models.yaml).
The official-source audit, gate status and weight-size evidence are retained in
[model-source-audit.json](environment/model-source-audit.json). Model-native
preprocessing and actual precision are recorded; this is not an experiment that
isolates architecture alone. Model weights are excluded from Git and ZIP.

Detailed official loading, licences and compatibility sources:
[retrieval and assistance](docs/PAIR1_PAIR3_SOURCES.md),
[generation and reconstruction](docs/PAIR2_PAIR4_SOURCES.md),
[dataset licences](docs/SOURCES_AND_LICENSES.md).

## Hardware and environment

Target: native Windows, NVIDIA RTX 4070 SUPER with 12 GB VRAM, Python 3.11 and CUDA
PyTorch. Actual hardware detection found RTX 4070 SUPER, 12,282 MiB total VRAM,
driver 610.88, and initially CPU-only torch 2.11.0. The driver CUDA capability in
`nvidia-smi` is not the CUDA build of PyTorch. See
[initial diagnostics](environment/doctor-initial-cpu.json) and
[current diagnostics](environment/doctor.json) for verified availability and free
memory at measurement time. Other desktop applications use part of the VRAM.
Approximately 385 GiB disk was free at initial inspection; recheck before downloads.

No driver, CUDA Toolkit, Visual Studio or system component was installed or
reconfigured. The initial environment had no project virtual environments. The
tested `.venv-retrieval` inherits existing system packages and adds local CUDA
wheels/dependencies; exact package versions and all inherited dependencies are
listed in environment/. This is an explicitly documented execution environment,
not a promise that arbitrary version combinations work. The setup script creates
clean isolated environments for reproduction. WSL status inspection returned
`Wsl/EnumerateDistros/Service/E_ACCESSDENIED`; WSL was not configured or tested.

## Install from Windows PowerShell

Run in the extracted project root. Python 3.11 must be available as `python`.
Do not use a global `pip install` for incompatible model stacks.

```powershell
python --version
nvidia-smi
# Core tools without inference dependencies:
.\scripts\setup_windows.ps1 -Pair core
.\.venv-core\Scripts\python.exe -m pytest -q
# Only after approval of CUDA and checkpoint download sizes:
.\scripts\setup_windows.ps1 -Pair retrieval -InstallCuda -ApproveLargeDownload
$python = '.\.venv-retrieval\Scripts\python.exe'
& $python -m stylesphere.cli doctor
```

The official CUDA wheel command is pinned to torch 2.11.0+cu128 and
torchvision 0.26.0+cu128. Explicit `+cu128` avoids accepting an already-installed
CPU wheel with the same public version. Python's `--system-site-packages` was used
for this development run only; the setup script's clean venv is preferred for
reproduction. Install all task requirements in separate environments. Optional 3D
rendering needs an appropriate OpenGL context and pyrender; core geometry metrics
require NumPy/SciPy and mesh loading requires trimesh.

For optional pinned reproduction, the observed Windows/Python 3.11 runtime locks
contain 52 retrieval and 46 assistance packages, including Torch's pure dependencies.
They exclude the separately installed Torch/torchvision binaries and unrelated
inherited packages. These are metadata receipts from the working interpreters,
not verified cold installations; see `environment/dependency-closure-retrieval.json`
and `environment/dependency-closure-assistance.json`. The following commands create
new environments and leave the measured environments intact. Downloads require
the size/disk checks described below.

```powershell
foreach ($pair in @('retrieval', 'assistance')) {
    $venv = ".venv-$pair-repro"
    if (Test-Path -LiteralPath $venv) { throw "Use a fresh environment: $venv" }
    python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    $taskPython = Join-Path $venv 'Scripts\python.exe'
    & $taskPython -m pip install --ignore-installed 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel installation failed' }
    & $taskPython -m pip install --ignore-installed --no-deps -r "environment/requirements-$pair-runtime.lock.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Observed runtime installation failed' }
    & $taskPython -m pip install --ignore-installed --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
}
```

Run each pair's CLI with its new `.venv-PAIR-repro\Scripts\python.exe` interpreter.
Do not replace a measured environment or interpret installing a lock as a completed
reproduction. `scripts/export_runtime_lock.py` regenerates these metadata-only
receipts when run explicitly in the corresponding interpreter.

For a download proxy that stalls on the full 2.75 GB CUDA wheel, the retained
`scripts/resume_cuda_wheel.py` uses bounded HTTP ranges and verifies the exact
official SHA256 before installation. Inspect its help and acquisition evidence;
do not bypass checksum verification. No toolkit reinstall is required to use
prebuilt PyTorch CUDA wheels.

## Checkpoints, gates and acquisition

Inference is **offline by default**. `prepare` does not download weights or large
datasets. `acquire` checks exact repository metadata, file sizes and disk space,
resolves a full commit and writes file SHA256s. It currently supports the approved
retrieval and assistance pairs, both gated 3D checkpoints and the Marqo tokenizer
dependency. Separate acquisition scripts select the exact generation components
and shared 3D auxiliary assets.

```powershell
& $python -m stylesphere.cli acquire --model fashion_siglip --dry-run
& $python -m stylesphere.cli acquire --model fashion_siglip
& $python -m stylesphere.cli acquire --model siglip2
& $python -m stylesphere.cli acquire --model fashion_tokenizer
```

Selected retrieval download sizes are approximately 816 MB, 1.54 GB and 2.45 MB,
respectively. The CUDA wheel is 2.75 GB. The user approved a 6-10 GB budget for this
setup plus these two models. The user subsequently approved another 50-60 GB for
the remaining official checkpoints and necessary isolated dependencies, subject
to disk checks and no driver/toolkit changes. Acquisition receipts record exact
selected files, source hashes and actual progress.
The retrieved configurations in `config/local.yaml` point to relative `models/`
directories; rerun acquisition after extraction because the ZIP omits weights.

Other model acquisition requires a separate size/space check and approval where
over 5 GB. Phi weights are about 7.67 GB; Qwen about 9.32 GB. The 4.07 GB FLUX checkpoint
also needs the official text encoder and VAE, so it is not a complete 4 GB pipeline.
SDXL needs both the selected UNet and compatible base components. Do not fetch
every checkpoint from the Lightning repository. SPAR3D and SF3D are gated; accept
Stability's terms with your own Hugging Face account. Never commit tokens. Their
custom CUDA/C++ extensions require a compatible existing compiler toolchain;
native Windows support is experimental. Both extensions were built and tested
using the existing toolchains; see [native 3D setup](docs/NATIVE_3D_SETUP.md).
Follow the separate official checkout instructions in [Pair 2/4 sources](docs/PAIR2_PAIR4_SOURCES.md). WSL2 is a documented
local-GPU fallback to investigate if native builds fail, with separate Linux
environments; no WSL success is claimed.

After accepting both gated repository licences in your browser, create a Read
token at https://huggingface.co/settings/tokens and sign in from your own terminal:

```powershell
hf auth login --force
# Paste the token at the hidden prompt; Git credential storage is unnecessary.
hf auth whoami
& $python scripts/check_gated_access.py
```

Never paste a token into chat, source code, configuration or command-line arguments.
The access checker records only gate status and revision, without account identity
or credentials. Licence acceptance belongs to the same account as the token.

For an interactive login followed by the two gate checks, run
`.\scripts\login_huggingface_local.ps1` in PowerShell. It uses the official hidden
token prompt and keeps the token in the normal Hugging Face user cache outside
this project. If the CLI still asks about Git credential storage, enter `N`.

The framework never sets `trust_remote_code=True`. Marqo uses reviewed OpenCLIP
code with a safetensors checkpoint, without executing the model repository's
Python wrapper. Qwen uses its integrated Transformers class. Exact FLUX FP8 uses
reviewed ComfyUI core in an isolated local subprocess; no server or custom nodes
are started and no hosted inference is used.

## Prepare and freeze datasets

```powershell
& $python -m stylesphere.cli prepare --all
# Acquire the tiny real8-image,4-query smoke catalogue separately:
& $python scripts/acquire_retrieval_smoke.py
# When the ZIP already contains the pinned manifest but omits licensed images:
& $python scripts/acquire_retrieval_smoke.py --restore-missing
& $python -m stylesphere.cli smoke --pair retrieval --config config/retrieval_smoke.yaml
```

The smoke catalogue has **no human relevance judgements**. It cannot be used with
`evaluate`; the smoke-only configuration rejects that command. Metrics without
positive judgements are null, never invented scores.

`--restore-missing` restores only omitted images from the existing manifest's exact
revision and SHA256s, with a 100 MB transfer cap. It preserves the manifest, catalogue,
queries and existing valid images. A changed/corrupt image, unavailable pinned
revision or hash mismatch fails explicitly; no newer source is substituted. Default
mode remains strict about missing images. This restore command was verified in an
isolated temporary directory: 8 images restored, 162,603 response bytes, all hashes
matched; the original smoke inputs and copied manifest remained unchanged.

The original targets were 600 product images / 120 queries, 24 briefs × 3 outputs
per model, 80 controlled assistance scenarios, and about 24 reference objects.
The [protocol amendments](docs/PROTOCOL_AMENDMENTS.md) distinguish these targets
from the actual sampling frame. The retrieval set contains 600 licensed ABO images
and 120 controlled queries: 240 GPU query executions succeeded and resumed without
rerunning. Original human relevance fields remain unjudged; the completed separate
[AI proxy analysis](results/retrieval/ai_assessment/v1/README.md) covers all 120
queries with uncertainty retained. The original reconstruction input set contains
20 real ABO objects without aligned reference meshes/cameras. A disjoint
[Google Scanned Objects extension](docs/REFERENCE_RECONSTRUCTION.md) supplies
20 test and 4 development reference objects. Dataset provenance, import formats and
exact commands are in [DATASET_PROVENANCE.md](docs/DATASET_PROVENANCE.md).

Run `scripts/acquire_abo_retrieval.py`, then `prepare --pair retrieval --config
config/abo_retrieval.yaml` to reproduce the 600-image subset. Use `evaluate --pair
retrieval --config config/abo_retrieval.yaml` for both model rankings. The source
sampling frame is one declared ABO metadata shard, with no apparel coverage.
The 120 query texts are explicitly hypothetical, AI-assisted controlled shopping
requests, rather than observed customer enquiries. No relevance label was derived
from either model's rankings. `scripts/export_relevance_review.py` creates a local worksheet
at datasets/retrieval/review/index.html from 3,600 seeded independent candidate pairs.
Grades begin blank; a human supplies 0-3 ratings and exports human_annotations.csv.
Import with datasets.retrieval.annotations_path in a new frozen config/version;
retain incomplete-pool limitations and do not turn unjudged products into negatives.

| File | Required record contents |
|---|---|
| `datasets/retrieval/catalogue.csv` | Stable product ID, local image path, attributes, category, source/revision/licence/hash |
| `queries.csv` | Query ID/text/category, qrels completeness flag |
| `qrels.csv` | Query/product ID, independently assessed 0-3 relevance, annotator/date |
| `datasets/generation/prompts.jsonl` | 24 hypothetical briefs with mandatory criteria and seeds |
| `datasets/assistance/knowledge_base.json` | 12 hypothetical products, prices/variants/stock and delivery/return policies |
| `scenarios.jsonl` | 80 requests, permitted read-only tools, visible records, exact required facts, expected outcome/category |
| `datasets/reconstruction/objects.csv` | Object/image/source, optional reference/cameras/preparation, explicit metric eligibility |

For retrieval, use real licensed Fashion200K/ABO assets and independent human
judgements. The candidate annotation pool is seeded without model scores. Retain
pool coverage/limitations. Fashion200K was used in Marqo's published evaluation;
it is contextual evidence rather than evidence of unseen-domain generalization.
Do not download full ABO or OmniObject3D archives. For 3D, start with shoes,
handbags and accessories; retain known cameras or controlled reference renders.
Geometry metrics require explicit alignment and coordinate units. Plausible
unseen surfaces do not establish physical accuracy.

Freeze inputs and protocol before final execution. Tune only on a separate
development subset. SHA256s, case IDs, checkpoint revisions, seeds and protocol
hashes tie outputs to the exact experimental context.

## Run pairs and complete assessments

Each command has `--help` and supports `--root`, `--config`, `--output-dir` and
`--seed`. Use the appropriate isolated interpreter. `--model KEY` can restrict a
pair to one named model for diagnosis or incremental execution under the same
frozen pair protocol.

```powershell
& $python -m stylesphere.cli smoke --pair retrieval
& $python -m stylesphere.cli evaluate --pair retrieval
.\.venv-assistance\Scripts\python.exe -m stylesphere.cli smoke --pair assistance --config config/assistance_smoke.yaml
.\.venv-assistance\Scripts\python.exe -m stylesphere.cli evaluate --pair assistance
.\.venv-generation\Scripts\python.exe -m stylesphere.cli smoke --pair generation --config config/generation_smoke.yaml
.\.venv-generation\Scripts\python.exe -m stylesphere.cli evaluate --pair generation
.\.venv-preview\Scripts\python.exe -m stylesphere.cli smoke --pair reconstruction --config config/reconstruction_smoke.yaml
.\.venv-preview\Scripts\python.exe -m stylesphere.cli evaluate --pair reconstruction
# Dispatch all task environments sequentially:
.\scripts\run_all_evaluations.ps1
# Same-interpreter form, only when all required task dependencies are available:
& $python -m stylesphere.cli evaluate --all
```

For rendered reconstruction previews, create `.venv-preview` using the
[documented overlay setup](docs/NATIVE_3D_SETUP.md#rendered-preview-environment)
and [six exact pins](requirements-preview.txt). The parent CLI renders views;
native model inference still uses the isolated `.venv-sf3d`/`.venv-spar3d` workers.
The [synthetic OpenGL smoke](environment/preview-context-smoke.json) passed on
RTX 4070 SUPER with driver 610.88; a separate cold rebuild has not been validated.

Default retrieval smoke uses the final catalogue and therefore blocks until it
exists; use `config/retrieval_smoke.yaml` for the included diagnostic catalogue.
Successful cases resume only when mode, protocol, configuration, inputs and saved
artifact hashes match. Failures/blockers are appended and never deleted.
Exit 0 means command execution succeeded, 2 means partial/blocked inference, and 1
means an unexpected command/integrity error. `verify` can pass integrity while
listing unfinished benchmark work; it does not certify assignment completion.

Shopping models see the same 5 read-only tools: search_products,
get_product_details, check_inventory, get_delivery_policy, get_return_policy.
Only the application validates and executes tool calls. The controller rejects
unknown arguments, hidden records, writes and external actions. Required gold
facts are withheld from prompts. Exact claim checks are separate from human prose
correctness; ambiguous answers remain human-review cases.
Shared controller protocol 2 also exposes canonical paths for facts already returned
by permitted tools. It adds no expected answer, hidden fact or automatic JSON repair.

Generation exports anonymous metadata-stripped images, `review.html`, blank
`ratings.csv` and a private mapping. Give raters only HTML/images/CSV; withhold
mapping until ratings are frozen. Ratings cover adherence, coherence, composition,
usability and a separate binary usable decision. Do not fill ratings automatically.
See [qualitative review protocol](docs/QUALITATIVE_REVIEW.md) and blank templates
under datasets/qualitative/. Human observations must identify rater and actual case.

The real SDXL 72-output run has a
[deterministic four-category QA gallery](results/generation/c72db9e579f046999779f6fba62b7849/qa/sdxl_lightning_first_per_category.png)
and explicitly labelled AI-assisted observations. These document visible examples,
including missing attributes and extra products; they contain no human ratings
and estimate no population error rate. Keep model-identifying QA material away
from blinded raters. After both models finish, run
`python scripts/export_generation_review.py` to create one canonical worksheet at
`results/generation/human_review`; withhold its private mapping and manifest.

## Metrics, statistics and generated evidence

[METRIC_DEFINITIONS.md](docs/METRIC_DEFINITIONS.md) defines nDCG@10, MRR, Hit@5,
conditional Recall@K, controlled grounded claim/tool metrics, prompt-clustered
human ratings, Chamfer, F-score and eligible camera/view comparisons. It also
defines timing windows, MiB units, missingness and bootstrap confidence intervals.
Default seed is 8410; bootstrap iterations 2000; generation repeats 3. Identical
seeds across different architectures do not imply identical latent noise.

Every successful output has a case record and artifact SHA256. Retrieval stores
embeddings, full rankings, scores and frozen manifest copies. Assistance stores
visible model turns, proposed/validated tool calls, observations, final response,
structured assessments and frozen business inputs. Internal thinking is excluded
from the scored visible response. Generation stores actual PNGs and exact frozen
prompts; audited sidecars bind the earlier SDXL records without rewriting them. Reconstruction
stores GLBs and eligible sampled surfaces, metric protocols and rendered views.

Model-load time, warmup policy, preprocessing/inference/end-to-end timings and
peak CUDA allocator memory are recorded where supported. Workers retain completed
cases if a later case fails. Out-of-memory, malformed outputs, export failures,
missing access/dependencies/data and invalid tool actions remain in the evidence.

## Regenerate, verify and package

```powershell
& $python -m stylesphere.cli report
& $python -m stylesphere.cli verify
.\.venv-preview\Scripts\python.exe scripts/finalize_evaluation.py
.\.venv-preview\Scripts\python.exe -m stylesphere.reporting.completion finalize
.\.venv-preview\Scripts\python.exe -m stylesphere.reporting.completion verify
& $python -m stylesphere.cli package
```

Reports read actual records and independently completed ratings. Outputs include
`results/reports/model_summary.csv`, failures.csv, execution_error_summary.csv, paired_comparisons.json,
human_generation_reviews.json, REPORT.md and hashed manifest.json. PNG/SVG charts
are created only if actual eligible **benchmark** values exist. Smoke diagnostics
appear in tables with explicit mode but never benchmark comparisons. Configurations
are not pooled across protocols. Prompt/object clusters preserve dependency in
paired bootstraps. Missing values stay missing; no winner is manufactured.

`results/assistance/human_assessments.csv` gains blank rows for successful benchmark
responses. Human raters identify themselves and assess task success, correctness
of prose and unsupported claims using the retained record hash. Completed numeric
assessments are summarized in `human_assistance_reviews.json`; blank rows are not
observations. Reports retain worksheet hashes, and verification detects changes
that require report regeneration.

Verification checks required files, input schemas, provenance flags, checkpoint
identities, artifact paths/hashes, retained ranking labels and recomputed metrics,
read-only shopping trace replay, valid images/meshes and report-source hashes.
It cannot cryptographically prove model origin merely from self-consistent logs;
retain live execution evidence and independently reproduce it.

The final ZIP is `submission/StyleSphere_Evaluation.zip`. Its manifest enumerates
file hashes, validated component completion, AI assistance and omitted restricted assets. It
excludes environments, caches, weights and recognizable credentials. Dataset
images/meshes require explicit per-record redistribution permission; otherwise
metadata and acquisition instructions are included. The original partial ZIP is a
historical snapshot. The legacy `benchmark_complete=false` field refers to the
original independent-human/absolute-reference requirements; the separately checked
`requested_component_complete=true` identifies completion of the confirmed scope.
AI proxy results do not populate original human fields. Report/video/presentation
production and three additional model pairs are outside this component's scope.

## Troubleshooting and limits

- CUDA unavailable: inspect interpreter path and `torch.version.cuda`; use the
  CUDA venv rather than replacing drivers. A CPU-only torch blocks inference.
- Missing local snapshots: run approved acquisition, inspect pinned paths and
  recorded hashes. Models are not silently downloaded during experiments.
- GPU out of memory: preserve the failed case; reduce documented batch size or
  use approved quantization/offload and freeze the resulting new configuration.
- Access gated: accept terms normally; do not bypass access controls or embed
  credentials. SF3D/SPAR3D require their own compiler/dependency setup.
- Missing relevance labels/reference alignment: report null quality metrics;
  collect independent annotations or eligible geometry rather than infer labels.
- Missing OpenGL: GLB success can remain valid while view metrics are explicitly
  unavailable. Use the documented renderer in a compatible local environment.
- Report source changed: rerun report before verify/package.

See [LIMITATIONS.md](docs/LIMITATIONS.md) for exposure, rights, fairness, privacy,
safety, statistical coverage and industry-readiness caveats, and
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for a concise marker checklist.
