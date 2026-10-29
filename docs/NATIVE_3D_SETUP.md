# Native Windows 3D deployment

Native builds, separate development smokes and both final 20-object computational
runs were executed on 29 October 2026. Initial authenticated requests returned
HTTP 401 / GatedRepoError; those historical diagnostics are retained. The user's
normal local login subsequently verified both gates, and both official main
checkpoints were downloaded and SHA-checked. No gate was bypassed or terms accepted
by the agent; no system components were installed. Computational completion does
not supply missing human quality judgements or reference geometry.

## Source, toolchain and tests

| Component | Actual evidence |
|---|---|
| SPAR3D source | `vendor/SPAR3D`, commit `fdc311b16809e6a8adc2f5a3407ebb3db1a95bd1`, tracked files clean |
| SF3D source | `vendor/SF3D`, commit `ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2`, tracked files clean |
| Existing native tools | VC 14.44.35207 activated from installed Build Tools; CUDA toolkit 12.6 selected explicitly |
| Isolated environments | `.venv-spar3d` and `.venv-sf3d`, sharing retrieval's PyTorch 2.11.0+cu128 |
| UV and texture extensions | Both environments built and imported both extensions |
| Numerical CUDA checks | Synthetic 16×16 triangle rasterisation/interpolation passed separately; not model inference or benchmark records |
| Initial model imports | Both official Python model classes first imported without loading weights |
| Actual model execution | Separate SF/SP development smokes, then 20 final cases each under the paired freeze; details below |

Actual evidence lives in `environment/build-*.log`, `native-build-*.json`,
`native-artifacts-*.json`, package snapshots and `synthetic-native-cuda-*.json`.
Unsuccessful activation, SDK-setting, unpatched compilation and dependency-import
attempts are retained. No driver, toolkit, compiler or Visual Studio installation
was performed. Existing CUDA 12.2/12.6/13.0 and VC 14.44/14.51 installations were
discovered despite `cl` being absent from the initial shell.

Builds used `DISTUTILS_USE_SDK=1`, `MSSdk=1`, `TORCH_CUDA_ARCH_LIST=8.9`,
`USE_NATIVE_ARCH=0` and two jobs. Compilation used CUDA 12.6 while PyTorch was built
for 12.8. Synthetic checks and the subsequent actual model runs passed on this
combination; that is evidence for this tested deployment, not arbitrary toolchain
compatibility. Optional `gpytoolbox` and `pynanoinstantmeshes` triangle/quad remeshing
packages were absent. The frozen `remesh="none"` path was used; no remeshing is claimed.

## Dependency isolation and include-only patch

For a **fresh extraction**, the following PowerShell commands prepare local-only
environments. They are concrete continuation commands, not a claim that a second
cold installation was executed. Use Python 3.11. Existing system compilers/toolkits
must already be installed; none of these commands installs or changes them.

Fetch the exact reviewed source commits into the fresh extraction:

```powershell
$sources = @(
  @('SPAR3D', 'https://github.com/Stability-AI/stable-point-aware-3d.git', 'fdc311b16809e6a8adc2f5a3407ebb3db1a95bd1'),
  @('SF3D', 'https://github.com/Stability-AI/stable-fast-3d.git', 'ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2'),
  @('OpenAI-CLIP', 'https://github.com/openai/CLIP.git', 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'),
  @('AlphaCLIP', 'https://github.com/SunzeY/AlphaCLIP.git', 'ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff')
)
foreach ($source in $sources) {
  $checkout = Join-Path 'vendor' $source[0]
  if (Test-Path $checkout) { throw "Preserve and inspect existing checkout: $checkout" }
  git init $checkout
  git -C $checkout remote add origin $source[1]
  git -C $checkout fetch --depth 1 origin $source[2]
  git -C $checkout checkout --detach FETCH_HEAD
}
```

Create one local shared CUDA environment. Reusing an already acquired local Torch
wheel avoids downloading it again; the official-index fallback below is for a new
machine/extraction and requires the documented download budget and disk check.

```powershell
python -m venv .venv-3d-shared
.\.venv-3d-shared\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
$root = (Get-Location).Path
$sharedSite = Join-Path $root '.venv-3d-shared\Lib\site-packages'
foreach ($model in @('sf3d', 'spar3d')) {
  python -m venv ".venv-$model"
  $interpreter = Join-Path $root ".venv-$model\Scripts\python.exe"
  & $interpreter -m pip install --ignore-installed --no-deps -r "environment/requirements-$model-runtime.lock.txt"
  $overlay = Join-Path $root ".venv-$model\Lib\site-packages\stylesphere_shared_torch.pth"
  [System.IO.File]::WriteAllLines($overlay, @($sharedSite, (Join-Path $root 'src')), [System.Text.UTF8Encoding]::new($false))
}
.\.venv-spar3d\Scripts\python.exe -m pip install --ignore-installed --no-deps --no-build-isolation vendor/OpenAI-CLIP vendor/AlphaCLIP
```

The locks include the observed runtime dependency closure (60 SF packages, 70 SPAR
packages), including pure dependencies needed by Torch. They exclude Torch/torchvision
binary packages and the two reviewed VCS installs. No global site-packages are added
to these fresh overlays. Flet GUI/CLI/server dependencies are intentionally excluded;
the headless inference worker only needs the imported controls/runtime. The closure
reports record that choice and distinguish observed versions from an unexecuted
cold-install verification. Local pins include SF tokenizers 0.19.1 and SPAR packaging
23.2/oauthlib 3.2.2/wheel 0.42.0, resolving constraints found during the audit.

Then run the native build commands below, followed by:

```powershell
.\.venv-sf3d\Scripts\python.exe scripts/record_3d_native_profile.py --model sf3d
.\.venv-spar3d\Scripts\python.exe scripts/record_3d_native_profile.py --model spar3d
```

This records the new local binary/build receipt hashes without changing official
source or checkpoint identities. Acquire the separately documented auxiliary/main
weights only after size/access checks. The tested workstation reused its existing
retrieval CUDA wheel; the fresh-extraction recipe keeps equivalent sharing entirely
within the extraction and does not rely on that workstation's global packages.

After acquisition, verify the local source, native binaries and initializer hashes
without loading a model or using CUDA:

```powershell
.\.venv-sf3d\Scripts\python.exe scripts/audit_3d_prerequisites.py --model sf3d
.\.venv-spar3d\Scripts\python.exe scripts/audit_3d_prerequisites.py --model spar3d
```

The earlier audit reports distinguish then-missing gated main files from verified
public initializers. Later acquisition receipts and actual model runs establish
the subsequent state; an initial prerequisite audit is not a model smoke result.

The official requirements pin Transformers 4.42.3, NumPy 1.26.4 and Hub 0.23.4.
Local site-packages precede `.pth` overlays exposing the acquired CUDA PyTorch and
inherited pure dependencies. Exact observed package versions are retained. A fresh
environment without those inherited packages needs the corresponding dependencies
explicitly installed. Use `--ignore-installed` and a local prefix for overlay
installations; explicit pinned components were installed with `--no-deps` so inherited
packages would not be uninstalled. Initial interrupted installers and safe local-only
installation logs remain available.

SPAR additionally uses reviewed OpenAI CLIP commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` and AlphaCLIP commit
`ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff`. Transparent-background 1.3.3 imports GUI
symbols even in library use. Flet/Core/Runtime 0.24.1 fixed its import against the
current incompatible Flet API. Albumentations 1.4.15 and Albucore 0.0.16 were used;
the offline worker disables the automatic update probe.

Unpatched texture compilation failed with MSVC C2872 (`std` ambiguous) in PyTorch's
`compiled_autograd.h`. Only the CUDA translation unit's unused Python-binding
header was replaced:

```diff
-#include <torch/extension.h>
+#include <torch/types.h>
```

Python bindings stay in the original `baker.cpp`; Tensor aliases, kernels,
architecture and final weights are unchanged. Overlays are under
`.cache/build-overlays/`; official source remains clean. The patch SHA256 is
`c59b6e46066c0f60cf9b566b14f963366b98cd39fd3a17af26f22eb29509667d`.
Receipts retain original/overlay hashes, compiler settings, build results and binary
hashes. The actually exercised build pattern is:

```powershell
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
python scripts/build_3d_native.py --model sf3d --cuda-path $cuda --vcvars $vcvars --cuda-header-compatibility
python scripts/build_3d_native.py --model spar3d --cuda-path $cuda --vcvars $vcvars --cuda-header-compatibility
```

Those paths describe this workstation's existing tools. The script requires explicit
existing paths and installs no system components. `config/reconstruction-local.yaml`
pins these exact binaries/receipts. A marker rebuilding locally must record the new
corresponding hashes; do not falsely mark an unreviewed checkout or build clean.

## Rendered preview environment

Use `.venv-preview` for the parent CLI when rendered reconstruction previews are
desired. Native model inference still runs in the configured `.venv-sf3d` and
`.venv-spar3d` worker interpreters. The preview overlay pins NumPy 1.26.4 because
PyRender 0.1.45 uses `np.infty`, which is absent from the inherited NumPy 2.4.2.

The following recreates the **observed overlay arrangement**, using an already
prepared `.venv-retrieval` and its read-only inherited dependencies. Its shared
pip/setuptools/wheel are used to build PyOpenGL's pure-Python wheel locally.
The six exact pins in [requirements-preview.txt](../requirements-preview.txt)
are an overlay, not a complete cold-install lock: the tested environment also
inherited SciPy 1.17.1, trimesh 4.12.2, Pillow, networkx and six. A separate cold
rebuild has not been validated. Run from the project root with Python 3.11:

```powershell
$previewRoot = (Get-Location).Path
$previewSharedPython = Join-Path $previewRoot '.venv-retrieval\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $previewSharedPython)) { throw 'Prepare the shared retrieval runtime first' }
if (Test-Path -LiteralPath '.venv-preview') { throw 'Preserve the existing preview environment; use a fresh extraction' }
& $previewSharedPython -m venv --without-pip --system-site-packages .venv-preview
if ($LASTEXITCODE -ne 0) { throw 'Preview environment creation failed' }
$previewPython = Join-Path $previewRoot '.venv-preview\Scripts\python.exe'
$previewSite = Join-Path $previewRoot '.venv-preview\Lib\site-packages'
$previewOverlay = Join-Path $previewSite 'preview-overlay'
$previewCache = Join-Path $previewRoot '.cache\pip-preview'
foreach ($directory in @($previewOverlay, $previewCache, (Join-Path $previewRoot 'environment'))) {
  New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$previewPaths = @($previewOverlay, (Join-Path $previewRoot '.venv-retrieval\Lib\site-packages'), (Join-Path $previewRoot 'src'))
[System.IO.File]::WriteAllLines((Join-Path $previewSite 'stylesphere-preview-runtime.pth'), $previewPaths, [System.Text.UTF8Encoding]::new($false))
& $previewPython -m pip --isolated --disable-pip-version-check --no-input --keyring-provider disabled install `
  --index-url https://pypi.org/simple --target $previewOverlay --ignore-installed --no-deps --no-build-isolation `
  --cache-dir $previewCache --report (Join-Path $previewRoot 'environment/preview-pip-reproduction-report.json') `
  -r (Join-Path $previewRoot 'requirements-preview.txt')
if ($LASTEXITCODE -ne 0) { throw 'Preview overlay installation failed' }
& $previewPython -m stylesphere.cli --help
```

The target directory precedes shared/inherited packages on `sys.path`; these
commands do not uninstall or replace them. The tested install used only official
PyPI artifacts (21,491,511 bytes, including the ZIP variant of PyOpenGL's source).
[The runtime receipt](../environment/preview-runtime-receipt.json) records exact
module origins, source hashes, the local wheel hash and CPU import/compatibility
checks; the original retrieval NumPy remained 2.4.2. The later
[OpenGL context smoke](../environment/preview-context-smoke.json) rendered a
synthetic 64×64 unit box on RTX 4070 SUPER, OpenGL 4.6.0 NVIDIA 610.88. That is a
renderer compatibility diagnostic, not a 3D model result. No system driver,
toolkit or Visual Studio component was installed or changed.

## Offline initializers and local execution

The exact public initializer plan is separate from gated main weights:

| Initializer | Immutable identity | Main weight bytes |
|---|---|---:|
| DINOv2-large, shared once | `facebook/dinov2-large`, `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` | 1,217,522,888 |
| SF OpenCLIP `.bin` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, `1a25a446712ba5ee05982a381eed697ef9b435cf` | 605,219,813 |
| SPAR OpenAI ViT-L/14@336px | Official URL SHA256 `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` | 934,088,680 |

`python scripts/acquire_3d_auxiliary.py` estimates/checks disk and uses two 8 MiB
range workers. OpenAI's URL/size/SHA are checked explicitly. OpenCLIP 2.24's local
loader requires its official `open_clip_pytorch_model.bin`; a safetensors file is
not silently substituted. Completion and actual hashes come from
`environment/acquisition_*.json`, not this size table. Auxiliary paths in the config
resolve relative to each `models/<model>` directory.

The worker checks final checkpoint size/SHA256 and original config Git-blob identity
against the frozen official acquisition plan, plus native receipt/patch/binary hashes
and the original source commit, then verifies local initializer bytes. The local
config retained those main-file identities before acquisition. An operational YAML changes initializer
paths only; unexpected official backbone/tag identities fail without substitution.
A Python audit hook rejects socket connections during inference, including
AlphaCLIP's separate downloader that ignores HF offline flags. Raw diagnostics
retain build and initializer provenance. Main weights still require the user's
normal Hugging Face acceptance and local authentication; no credentials are logged.

```powershell
.\.venv-preview\Scripts\python.exe -m stylesphere.cli smoke --pair reconstruction --config config/reconstruction_smoke.yaml
.\.venv-preview\Scripts\python.exe -m stylesphere.cli evaluate --pair reconstruction --config config/reconstruction-local.yaml
```

The CLI dispatches each isolated model interpreter. These commands were exercised
with the acquired local weights on the tested workstation. Fresh extraction still
requires the documented sources, build receipts, approved weight acquisition and
dependencies; a second cold rebuild has not been validated.

## Executed development and final reconstruction

Official SF/SP `model.safetensors` payloads total 11,351,239,332 bytes, with source
SHA256 checks in `environment/acquisition_sf3d.json` and `acquisition_spar3d.json`.
The verified public initializers are separate. The preview overlay acquired
21,491,511 bytes of official PyPI artifacts; driver 610.88 and existing system
toolchains remained unchanged. No cloud inference was used.

The development SF run `6cbc66f903e545ada9f04d06bb49cdf6` exported a valid GLB
and four RGB/mask view pairs: inference 0.6896215s, constructor load 9.5671108s,
allocated/reserved peaks 6128.982910/7336 MiB. SP run
`beffe85dbbc14673babc7b7a0a9822b8` also exported a valid GLB and four view pairs:
inference 14.1526668s, constructor load 0.6667598s, worker E2E 14.2411757s,
excluded warmup 20.2620753s, allocated/reserved peaks 8019.16455/8460 MiB.
These development cases are separate from the final 20-object set.

Before final inference, sources, settings, dependencies and inputs were frozen in
[the deployment receipt](../environment/reconstruction-deployment-freeze.json):
freeze `217deb592939b3aed0c49201b4955a9026b0f0d2bdf8644649f5998ef9d5c48a`,
protocol `59d2dda9fd312f0303a399fc6615bd1ae2b462ab6554ae3e51ef296b055b21ba`.
Both final runs used identical 20 case IDs and seed 8410, sequentially on local CUDA.

| Model / final run | Exports | Strict-valid meshes | Mean inference | Mean worker E2E | Max allocated / reserved |
|---|---:|---:|---:|---:|---:|
| SF3D `7685fb6f464d4360a1ebadb8b1994d49` |20/20|19/20|0.67775616s|0.75306838s|6128.982910/7342 MiB|
| SPAR3D `d088458f86e447f58e6c81f257cbb027` |20/20|20/20|12.70919922s|12.78761729s|8021.246582/8484 MiB|

Both runs had zero failed/blocked executions. SF `abo-B07XPHD4LK` has one
degenerate face among 17,152: export and four previews succeeded, but strict
`mesh_valid=0`. The defect was retained. All 40 final objects have four RGB views
and four masks (160 of each). Both models have 0/20 watertight outputs under the
exported indexed topology; texture UV seams can split coincident vertices, so a
false flag alone does not establish missing geometry. Neither export nor strict
mesh validity establishes reference reconstruction accuracy.

Both calls used verified CUDA BF16 autocast and the official native checkpoint;
no complete parameter-dtype distribution was measured. SP's official low-VRAM
mode defers module loading/swapping into `run_image`, inside `inference_s` and
worker E2E. Its inexpensive constructor is not evidence that all modules were
resident before inference. Worker E2E excludes parent-process mesh validation
and preview rendering. Peak counters are unmodified PyTorch allocator MiB, not
measurements of physical device residency.

[Resume evidence](../environment/reconstruction-resume-check.json) records all
40 cases resumed with no new inference and unchanged raw-record SHA256. The
independent [freeze audit](../environment/reconstruction-freeze-integrity.json)
passed. Human visual ratings and matching reference geometry/cameras remain
unavailable; no accuracy score or automatic model winner follows from these runs.

## Real inputs and missing reference accuracy

The final 20 inputs contain 5 BOOT, 5 SHOES and 10 HANDBAG records from the frozen
600-product retrieval subset, drawn from official ABO listings_0 only. Candidate
order, seed and QA thresholds were saved before 3D model outputs. Twenty-one
candidates were screened; one handbag failed the major disconnected-alpha-component
rule. These are deterministic mask checks, not human ratings. The source-image
transfer was 9,068,932 bytes. Original hashes/S3 keys/response metadata and exclusions
are retained. Smoke commands use only the first boot row from the development
manifest; that boot is absent from the final 20. Its second handbag row is not
executed by those commands.

Common CPU preparation uses rembg 2.0.57 / U2NetP (4,574,861 bytes), verified against
the official rembg MD5 and retained SHA256. Photos are resized preserving aspect
ratio to a maximum side of 1024, then given predicted transparent alpha. Both
models receive identical prepared RGBA files; subsequent native framing defaults
remain model-specific. Human input-quality review is pending.

The selected official S3 assets use the saved current CC BY 4.0 licence.
`RIGHTS_AND_ATTRIBUTION.json` credits Amazon.com/ABO creators and records resize/
matting changes. Explicit redistribution covers the selected originals and RGBA
derivatives, not model weights. Initial conservative manifests were archived when
that rights record was added; selection and pixels did not change.

These fashion listings lack linked reference geometry and calibrated cameras.
All geometric/silhouette/reference-render eligibility flags are false. Actual
inference can support validity/export/time/memory comparisons, while Chamfer,
F-score and matching-view accuracy remain not computable. Household/furniture
meshes were not mislabelled as fashion ground truth. OmniObject3D needs separate
authorised partial-data access; public-document credentials were not reused.
