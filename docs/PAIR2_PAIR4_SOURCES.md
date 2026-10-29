# Generation and reconstruction: sources, operation and current limits

Official documentation/source was inspected on 29 October 2026. These adapters
are implemented. Numerical and review unit tests run locally. The initial
CPU-only environment was followed by a separately installed CUDA environment;
current execution status is established by actual `results/*/records.jsonl`,
not the presence of adapter code or acquired weights. Human ratings remain blank
until supplied by evaluators.

Subsequent authorized acquisition is recorded in `environment/acquisition_*.json`;
those receipts and actual inference records supersede this initial implementation
snapshot. Generation overlays `.venv-generation` and `.venv-comfy` share the
already installed project CUDA runtime through a `.pth` entry, with isolated local
dependency overlays and access to the pre-existing base Python packages. This
avoids a second multi-gigabyte CUDA wheel. They are not hermetic environments;
local and effective versions are recorded in `environment/`. Diffusers 0.37.0 is
pinned because 0.42.1 selected Hub 2.x, which conflicts with Transformers 5.5's
Hub <2 requirement. No drivers or system toolchains are altered.

Authorized acquisition uses `scripts/acquire_generation.py`. It first verifies
free space and the selected official sizes (19,393,067,944 bytes total), then
resumes only the pinned files in `models/hub`. It excludes the replaced SDXL
base UNet, BF16 FLUX and unused FP4 encoder. All large files are checked against
their official LFS SHA256 before publication; complete receipts are written under
`environment/acquisition_*.json`. A bounded HTTP Range transport handles the
observed proxy stalls and preserves completed bytes and journals across retries.
The generation helper requires prior approval for these downloads; it does not
authorize them itself. `config/generation-local.yaml` records the exact approved
asset revisions and reviewed ComfyUI commit. Root configuration must merge those
settings before offline inference.

## SDXL-Lightning

The adapter follows ByteDance's four-step UNet example: the exact
`sdxl_lightning_4step_unet.safetensors`, SDXL Base 1.0, FP16,
`EulerDiscreteScheduler(timestep_spacing="trailing")`, four inference steps and
guidance zero. CPU model offloading is enabled by default for the 12 GB target.
The checkpoint uses OpenRAIL++ terms; review the base model terms too.
[Official model card and loading example](https://huggingface.co/ByteDance/SDXL-Lightning).

Weights/configurations are resolved from the local Hugging Face cache. No
`from_pretrained` call in this adapter is allowed to fetch missing assets.
Set `models.sdxl_lightning.base_revision` and `revision` to fixed commits after
approved acquisition. The resolved Lightning snapshot revision is recorded.

## FLUX.2 Klein 4B FP8: exact format matters

The selected BFL repository contains a single ~4.07 GB
`flux-2-klein-4b-fp8.safetensors` diffusion checkpoint; it is not a complete
Diffusers pipeline. Loading the BF16 pipeline with a different dtype does not
establish use of these FP8 weights. Its licence is Apache 2.0. The card's VRAM
statement is contextual and is not a measurement from this workstation.
[Official FP8 repository](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/tree/main).

The implemented path uses reviewed **ComfyUI core** loaders in an isolated local
Python worker: UNETLoader with `weight_dtype="default"` preserves the checkpoint's
native quantization handling, CLIPLoader with `type="flux2"`, and VAELoader.
Generation uses four steps, Flux2Scheduler, Euler, BasicGuider (the guidance
distilled path), EmptyFlux2LatentImage and VAEDecode. No custom nodes or server
are initialized. The official ComfyUI tutorial specifies the distilled FP8 file,
`qwen_3_4b.safetensors` text encoder and `flux2-vae.safetensors` in separate model
folders. The complete asset set is larger than the diffusion checkpoint alone;
estimate all files before acquiring them.
[Official ComfyUI guide](https://docs.comfy.org/tutorials/flux/flux-2-klein).

Core interfaces inspected:
[loaders](https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py),
[Flux2 scheduler/latent](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_flux.py),
[guider/sampler](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_custom_sampler.py).
These are external APIs and may change; pin and record the checkout commit.
The reviewed ComfyUI source is GPL-3.0; retain its source licence separately
from the Apache-2.0 model-weight licence.
[ComfyUI source licence](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE).
Set optional `source_revision` to its exact Git commit; the worker then rejects
tracked modifications or a different commit. Actual commit/dirty status is
recorded separately from model-weight revision.
Runtime records also retain observed diffusion parameter dtype counts and
raw quantized payload dtypes, Comfy Kitchen backend availability, and passive
cumulative native FP8 matmul dispatch attempts/successes/failures across warmup
and measured inference. These counters observe the original operation and
rethrow its original exception; the library's backend/fallback policy is
unchanged. A dispatch failure can trigger the library's dequantization fallback
and must be disclosed. QuantizedTensor parameters expose their original logical
dtype, so a BF16 parameter wrapper does not establish BF16 weight storage;
the retained raw payload dtype identifies the underlying FP8 data. FP8 describes
the selected diffusion checkpoint, not every tensor in the text encoder, VAE or
pipeline. The current CUDA 12.8 wheel causes this
ComfyUI revision to disable its optional CUDA 13 optimized backend and retain the
supported eager/PyTorch backend. Loading the exact FP8 checkpoint does not imply
that every arithmetic operation uses FP8 tensor cores; no such claim is made.
The native BFL repository's BF16 loader is not substituted for scaled FP8.
[BFL model loader](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/util.py).

Configure these keys under `models.flux2_klein` (paths relative to project root
or absolute; do not put credentials here):

```yaml
checkout: external/ComfyUI
python: .venv-comfy/Scripts/python.exe
checkpoint: external/ComfyUI/models/diffusion_models/flux-2-klein-4b-fp8.safetensors
text_encoder: external/ComfyUI/models/text_encoders/qwen_3_4b.safetensors
vae: external/ComfyUI/models/vae/flux2-vae.safetensors
```

The last three paths default to the official filenames below the checkout.
Only the selected distilled 4B FP8 filename is accepted; Base/9B/BF16 substitutes
are rejected. A filename alone is not identity proof: record the approved source
revision and verify acquired file hashes. The separate development smoke
`f0d642cfeda84af7b62d3c979d8b022a` established compatibility on the RTX 4070 SUPER:
640 native FP8 matrix dispatches across warmup and inference succeeded, with no
dispatch failure. This is compatibility evidence, not a human quality grade or
the full benchmark result. Actual evaluation records determine final coverage.
Isolate ComfyUI dependencies from Diffusers and 3D environments.

## SPAR3D and Stable Fast 3D

Both checkpoints require the user to accept Hugging Face conditions before
access. Stability's Community licence and commercial limits must be reviewed
before business use. The project does not bypass gates.
[SPAR3D model card](https://huggingface.co/stabilityai/stable-point-aware-3d),
[SF3D model card](https://huggingface.co/stabilityai/stable-fast-3d).

Official native Windows support is experimental and lists Visual Studio 2022
with compatible PyTorch/CUDA. Both projects install local `texture_baker` and
`uv_unwrapper` extensions. SPAR3D's documented low-VRAM path is
`SPAR3D_LOW_VRAM=1`/`--low-vram-mode`; this adapter sets the variable and passes
`low_vram_mode=True` to the official model API. Published VRAM figures are not
local measurements.
[SPAR3D installation and low-VRAM instructions](https://github.com/Stability-AI/stable-point-aware-3d),
[SF3D installation](https://github.com/Stability-AI/stable-fast-3d).

The baker build selects PyTorch CUDAExtension when CUDA is available; the UV
unwrapper builds a C++ extension. These need compilers beyond a CUDA-enabled
PyTorch wheel. Do not install or change drivers, CUDA Toolkit or Visual Studio
automatically. If a reviewed native build fails, save its diagnostics and use a
separate WSL2 environment with NVIDIA GPU access, compatible CUDA compiler and
Linux build tools; do not mix Windows and Linux wheels or interpreters.
[Texture baker build](https://github.com/Stability-AI/stable-point-aware-3d/blob/main/texture_baker/setup.py),
[UV unwrapper build](https://github.com/Stability-AI/stable-point-aware-3d/blob/main/uv_unwrapper/setup.py).
WSL2 has not been executed or validated in this run.

Use an independent environment for each official checkout because its pinned
Transformers/NumPy dependencies can conflict with generation/assistance.
[SPAR3D requirements](https://github.com/Stability-AI/stable-point-aware-3d/blob/main/requirements.txt),
[SF3D requirements](https://github.com/Stability-AI/stable-fast-3d/blob/main/requirements.txt).
After user-approved toolchain setup, the official PowerShell installation pattern
is `Set-Location <reviewed-checkout>` followed by environment activation,
`python -m pip install setuptools==69.5.1 wheel`, and
`python -m pip install -r requirements.txt`. These commands were researched,
not executed. Review the dependency sources before installing.

For each model configure `checkout`, optional isolated `python`, optional local
`weights_path` containing `config.yaml` and `model.safetensors`, and pinned
`revision`. The worker uses the official SPAR3D/SF3D `from_pretrained`, native
foreground preparation, `run_image(..., remesh="none")`, and GLB export interfaces.
[SPAR3D reference orchestration](https://github.com/Stability-AI/stable-point-aware-3d/blob/main/run.py),
[SF3D reference orchestration](https://github.com/Stability-AI/stable-fast-3d/blob/main/run.py).

Supply the same previously prepared transparent RGBA object image to both models;
opaque images fail explicitly. The adapter avoids automatic background-model
downloads and preserves each model's documented foreground framing defaults
(SPAR3D crop ratio 1.3; SF3D foreground ratio 0.85). Record any preparation and
its source independently. The exact original source and prepared input must be
retained. CUDA is required explicitly, preventing official scripts' CPU fallback.

The offline initializer checks the official DINOv2 identity before redirecting
both SPAR3D's reconstruction image tokenizer and its independently constructed
point-diffusion image tokenizer to the same hash-verified local DINOv2 snapshot.
Only initializer paths change; tokenizer classes and architecture settings stay
as published in the pinned config. Each case resets Python, NumPy, Torch and CUDA
seeds immediately before its measured `run_image`, after any discarded warmup,
so warmup RNG consumption does not determine the measured first-case output.
This is a seed-control policy rather than a claim of cross-device bitwise
determinism. New 3D records retain the exact input descriptor as a hashed JSON
artifact, plus actual worker/runner source hashes and installed library versions
for binding independent development smoke runs to the deployment freeze.

## Measurements, metrics and human review

Each worker loads once per model. `evaluation.warmup_runs` defaults to one;
warmup output is discarded and excluded from benchmark records. Per-image
`inference_s` uses synchronized wall-clock time for prompt encoding/generation/
decode, or reconstruction's official `run_image`. File export is excluded;
3D export time is separate. Model load time is separate. Peaks are PyTorch
allocated/reserved MiB across the timed inference window, including existing
tensor allocations; these do not measure physical device residency. Unavailable measurements remain
missing. The first measured case can still have residual caching effects.

For both 3D models, `preprocessing_s` measures the case's initial seed setup,
source image read/validation, official foreground framing and prepared PNG save.
Worker `end_to_end_s` measures the wall interval from the start of that work to
completed GLB export, subtracting the explicitly measured warmup loop and its
synchronization interval (`warmup_excluded_s` when warmup occurs). It includes
measured inference, export and intervening case overhead; it excludes model
loading, later mesh validation, reference metrics and preview rendering in the
runner. Failure records retain the observed partial preprocessing/end-to-end
window up to the exception; an interrupted warmup's measured interval is still
subtracted. Unreached inference/export stages have no invented zero timings.

The verified 3D precision setting is CUDA BF16 autocast with native checkpoints;
no complete parameter-dtype distribution was measured. SPAR's official low-VRAM
path performs deferred module loads/swaps inside `run_image`, so those operations
are timed as inference/worker E2E. A low constructor `model_load_s` does not imply
that all modules were already resident. Optional `gpytoolbox`/`pynanoinstantmeshes`
were absent and the frozen `remesh="none"` path was used. Actual development/final
execution, strict mesh-validity exceptions and resume evidence are indexed in
[the native timeline](NATIVE_3D_SETUP.md#executed-development-and-final-reconstruction).

`evaluation.generation.repeats` defaults to three, width/height to 1024. Seeds are
the CLI seed plus repetition index, shared across both models. Smoke selects one
prompt and one seed. Identical integer seeds do not imply identical initial
noise distributions across different architectures.

Blinded review copies remove PNG metadata and use opaque filenames. Evaluators
receive `review.html`, `ratings.csv`, and `images/`; withhold `private_mapping.json`
until ratings are frozen. CSV ratings start blank and are never invented or
overwritten. Five-point anchors and a separate binary usable decision appear in
the worksheet. The predeclared primary `usable_concept_rate` takes an image's
unanimous nonblank binary usability decision; one rater counts, and any multiple
rater disagreement stays missing pending documented human adjudication. Decided
images average within prompt and then prompts within model. The supplementary
`usable_vote_rate` averages individual reviewer votes within image, then follows
the same prompt clustering. It must not be labelled an approved-concept rate.
Coverage reports total, voted, decided, disagreeing and missing-vote images per
model and prompt. Missing ratings remain missing. Preserve original rating files
and the audit trail if adjudication later resolves a disagreement. Partial rating
coverage must be reported, including whether missingness is systematic.

After both generation benchmarks finish, run
`python scripts/export_generation_review.py`. The canonical worksheet at
`results/generation/human_review` selects the latest genuine successful
evaluation records from both models under one matching generation protocol;
development smoke outputs are excluded. If multiple protocols qualify, explicitly
select `--protocol-sha256`. Its manifest binds original image SHA256 values,
record identities, the private mapping and the blinded PNG copies. Keep the
manifest and private mapping away from raters because they reveal model identity.
Run-specific blank sheets remain diagnostic exports. The canonical export can
rebuild an entirely blank worksheet, but once any rater identity, grade or note
exists it refuses changed selections, mapping or images and never rewrites the
worksheet. Reports use a read-only integrity check before aggregating it.

New generation runs retain exact `evidence/prompts.jsonl` bytes as a
`frozen_prompts` artifact with SHA256. Verification checks the entire matching
prompt-row digest against `input_sha256`, plus the recorded prompt and mandatory
criteria. The earlier SDXL development and full runs retain immutable raw records
and audited `evidence/prompt_binding.json` sidecars binding their original record
digests to exact frozen prompt bytes. The full run's frozen input SHA also matches
its pre-run deployment freeze. Such sidecars disclose their post-run audit time;
they do not imply that additional artifacts existed during inference.

Failed timed inference attempts retain their actual synchronized elapsed window
and observed allocator peaks, including CUDA OOM diagnostics. If synchronization
or a memory getter fails, the diagnostic is retained and unavailable measurements
remain missing. Model load/warmup failure windows are identified separately from
measured case attempts. Throughput denominators must include all measured
successful and failed case inference windows, not successful cases alone.

The 3D worker applies the same rule to failed CUDA windows: elapsed time and
allocator peaks survive the original exception, including OOM classification and
any synchronization/memory-getter error. Load and warmup failures retain their
own diagnostic windows and never become `inference_s`. An export failure keeps
the completed inference window plus the partial export time. Previously completed
cases and any partial exported/prepared files remain available for the runner's
artifact hashing; a later cleanup error cannot erase them. Existing raw records
are preserved without retroactively inventing missing historical measurements.

Surface samples use triangle area weights and uniform barycentric draws.
Chamfer is half the sum of mean squared nearest-neighbor distances; `chamfer_l1`
also records the unsquared alternative. F-scores use predeclared distance
thresholds in the declared common units. `preparation_metadata` must include an
explicit affine `alignment_transform` and `coordinate_units`; labels alone do
not establish alignment. With `normalization: shared_reference_unit_diagonal`,
both already aligned meshes receive the same reference-bounding-box translation
and diagonal scale; thresholds then use reference-diagonal units. This transform
is used only for geometry scores and never modifies camera-matched rendering.
With `normalization: none`, thresholds use the original coordinate units.
No implicit ICP/axis fitting is performed. Store the
alignment rationale before comparing final outputs.

Successful cases resume only when mode, model, input hash, seed, configuration
and protocol match and every retained artifact still has its recorded hash.
Smoke cases cannot satisfy full evaluation. Failed or blocked cases are retried.

`camera_info` supports JSON `views`, each with `camera_to_world` (OpenGL camera
axes), `width`, `height`, and `intrinsics` (3x3) or `yfov`; optional
`reference_silhouette`/`reference_rgb` paths permit eligible comparisons. Silhouette
IoU needs matching masks/cameras; both empty masks produce missing, not a perfect
score. Render similarity is normalized RGB MAE and PSNR, supplementary pixel
agreement, not perceptual or physical accuracy. Views aggregate within an object.

Optional pyrender standard previews use four azimuths, 20-degree elevation,
fixed lighting and white background. Preview-only centering/scaling never feeds
geometry scores. Missing OpenGL/render dependencies are recorded as skipped view
metrics while retaining valid mesh success. Its display/EGL/OSMesa requirements
depend on the platform.
[Official offscreen rendering documentation](https://pyrender.readthedocs.io/en/latest/examples/offscreen.html).
