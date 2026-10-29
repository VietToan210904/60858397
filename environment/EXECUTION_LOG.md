# Execution evidence index

Date:29October 2026 (Australia/Sydney). JSON timestamps use UTC.

| Action actually run | Stored evidence |
|---|---|
| Initial environment inspection | doctor-initial-cpu.json |
| Original CUDA wheel install/slow transfers, resumed verified installation | cuda-wheel-resume.jsonl, cuda-wheel-source.json, cuda-local-install.log, cuda-install-check.json |
| Selected checkpoint acquisition | acquisition_fashion_siglip.json, acquisition_siglip2.json, acquisition_fashion_tokenizer.json |
| Official identities/access/size audit for8models | model-source-audit.json |
| Prepared all4input schemas/controlled corpora | datasets/*/manifest.json |
| Initial CPU-only smoke/evaluation preflight | smoke-initial-cpu.log, evaluation-initial-preflight.log; retained case blockers in results |
| Current CUDA doctor after final GPU runs | doctor.json, doctor-final.log; earlier doctor-cuda.log retained |
| Genuine GPU retrieval smoke | retrieval-smoke-cuda.log;8success records + artifacts in results/retrieval |
| Successful resume without re-executing models | retrieval-smoke-resume.log |
|600ABOimages/120controlled queries:240successful query executions and240resumed | retrieval-abo-inference.log, retrieval-abo-resume.log, retrieval-abo-summary.json |
| SDXL-Lightning development smoke, then72benchmark outputs | sdxl-lightning-dev-smoke.log, sdxl-deployment-freeze.json, sdxl-benchmark-validation.json; results/generation |
| Phi BF16development: original protocol failure retained, revised shared observation tested | phi-development-smoke-20261028.log; results/assistance records and versioned traces |
| Both shopping benchmarks:160single-pass attempts,73strict-protocol failures retained | assistance-deployment-freeze.json, assistance-benchmark-summary.json, assistance-final-failure-diagnostics.json, *-final-evaluation-integrity.json |
| Exact FLUX FP8development and72benchmark outputs:0failures | flux2-deployment-freeze.json, flux2-benchmark-validation.json; results/generation |
| Combined144-image blinded worksheet:allratingsblank | results/generation/human_review/manifest.json |
| Both native3Dextensions built/imported and synthetic CUDAkernels checked | native-build-*.json, synthetic-native-cuda-*.json, docs/NATIVE_3D_SETUP.md |
| Other-pair smoke preflight in CUDA environment | assistance-smoke-cuda.log, generation-smoke-cuda.log, reconstruction-smoke-cuda.log |
| Automated test suite | tests-final.log |
| Exact package versions | packages-retrieval.json |
| Reports/integrity checks | results/reports/manifest.json, results/verification.json |
| ZIP byte-level validation | submission/package-result.json; SUBMISSION_MANIFEST.json |
| Fresh extraction and all-member hash verification | package-extraction-validation.json (identifies the exact checked ZIP SHA) |

Commands to repeat are in README.md. Only listed executed operations are claimed
successful. Real campaign images and development shopping responses have been
observed. No user-feedback session, human relevance assessment, human visual/prose
rating or reconstructed AI mesh has been observed at this snapshot.


## 29 October 2026 (Australia/Sydney): final reconstruction computational runs

This dated addition supersedes the earlier snapshot's statement that no actual
reconstructed mesh had been observed. Historical diagnostics and sections above
remain unchanged.

- The user's normal official local Hugging Face login verified both gates.
  Both main checkpoints were acquired and SHA-checked: 11,351,239,332 bytes of
  `model.safetensors` payloads in total. Public initializers were checked
  separately. Source identities and acquisition receipts are retained; no gate
  bypass, credential logging, cloud inference or system-component installation
  is claimed.
- Both separate development smokes succeeded before final execution. SF run
  `6cbc66f903e545ada9f04d06bb49cdf6` measured inference 0.6896215s and allocated/
  reserved peaks 6128.982910/7336 MiB. SP run
  `beffe85dbbc14673babc7b7a0a9822b8` measured inference 14.1526668s, constructor
  load 0.6667598s, worker E2E 14.2411757s, excluded warmup 20.2620753s and peaks
  8019.16455/8460 MiB. Each exported a valid GLB and four RGB/mask preview pairs.
- The pre-final paired receipt `reconstruction-deployment-freeze.json` binds
  sources, settings, dependencies and final inputs. Freeze:
  `217deb592939b3aed0c49201b4955a9026b0f0d2bdf8644649f5998ef9d5c48a`.
  Shared protocol:
  `59d2dda9fd312f0303a399fc6615bd1ae2b462ab6554ae3e51ef296b055b21ba`.
  Both models then processed the same 20 case IDs with seed 8410, sequentially.
- SF final run `7685fb6f464d4360a1ebadb8b1994d49`: 20/20 execution and GLB-export
  completions, zero failed/blocked. Mean inference 0.677756159994s; mean worker
  E2E 0.753068380005s; maximum allocated/reserved 6128.982910/7342 MiB.
  Strict mesh validity was 19/20. `abo-B07XPHD4LK` retains one degenerate face
  among 17,152 while still exporting and rendering successfully; no repair or
  relabelling was performed.
- SP final run `d088458f86e447f58e6c81f257cbb027`: 20/20 execution/export
  completions and 20/20 strict-valid meshes, zero failed/blocked. Mean inference
  12.709199215s; mean worker E2E 12.787617285s; maximum allocated/reserved
  8021.246582/8484 MiB. All 40 final outputs have four RGB views and four masks
  (160 RGB and 160 masks). Both models have 0/20 watertight flags under exported
  indexed topology; split UV-seam vertices mean this alone does not establish
  missing geometry.
- Verified execution used CUDA BF16 autocast and official native checkpoints.
  No complete parameter-dtype distribution was retained. SP's official low-VRAM
  path defers module loads/swaps into `run_image`; these operations are inside
  inference/worker E2E, so cheap constructor timing does not establish that all
  modules were resident. Worker E2E excludes later parent-process mesh checks
  and preview rendering. Optional gpytoolbox/pynanoinstantmeshes were absent;
  the frozen remesh="none" path used neither triangle nor quad remeshing.
- `reconstruction-resume-check.log` reports all 40 cases resumed with no new
  inference. Its JSON receipt confirms raw records were unchanged, SHA256
  `c90b97c30eb1149626e941e3457e5313f87c5191b02280feec70c6b40d6de001`.
  Independent `reconstruction-freeze-integrity.json` passed. The retained
  `reconstruction-benchmark-validation.json` and its log index artifact/mesh
  validation; raw records and hashes remain authoritative.
- Preview's six-pin local overlay used official PyPI artifacts (21,491,511 bytes),
  with source/install/build/runtime evidence in `preview-runtime-receipt.json`.
  The synthetic OpenGL receipt records RTX 4070 SUPER, OpenGL 4.6.0 NVIDIA 610.88.
  Existing driver/toolkit/compiler/Visual Studio components were unchanged.
  Tested local installs/runs do not establish a separately validated cold rebuild.
- The pre-final full suite passed 125 tests with one optional OmegaConf skip;
  native focused tests passed 26 in SF and 3 in SP separately. See
  `tests-prerun-all3d.log` and the corresponding native test logs.
- Fresh report/verification checked 928 raw records, generated 274 summary rows
  and 30 paired comparisons, and reported zero integrity errors. Its 524 pending
  entries include historical blockers and missing human/reference evidence;
  they do not mean 524 unexecuted tasks. All 584 planned technical attempts are
  recorded: 240 retrieval + 144 generation + 160 assistance + 40 reconstruction.
  There are 511 technical completions and 73 strict assistance-protocol failures.
- Human relevance/visual/prose grades remain unfilled and matching 3D reference
  geometry/cameras remain unavailable. Export/mesh checks do not establish true
  reconstruction accuracy or a model winner. The package remains partial, with
  AI assistance disclosed and the requested four pairs/eight models distinct
  from the brief's seven pairs/fourteen models. This addition makes no claim
  that a refreshed ZIP has already been built.
