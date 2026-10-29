# Sources, licences and attribution

Sources checked on 29 October 2026. This is a record of official source claims and
acquisition requirements, not permission to redistribute every underlying asset.
Retain the licence text and exact revision alongside each acquired dependency.
Do not include credentials, gated weights, third-party image archives or caches in
the submission by default. Model code and model weights may have different licences.

## Datasets

| Source | Official licence statement | Attribution / qualifications |
|---|---|---|
| [Marqo Fashion200K](https://huggingface.co/datasets/Marqo/fashion200k) and [original Fashion200K repository](https://github.com/xthan/fashion-200k) | Apache-2.0 is declared by the repositories | Cite Han et al., *Automatic Spatially-aware Fashion Concept Discovery*, ICCV 2017. Images originated from Lyst; the mirror disclaims ownership. Do not assume the software/repository label resolves every retail-image right. |
| [Amazon Berkeley Objects / AWS Registry](https://registry.opendata.aws/amazon-berkeley-objects/) | CC BY 4.0 for all assets; registry notes 2023 change from CC BY-NC | Credit Amazon.com for assets and cite Collins et al., *ABO: Dataset and Benchmarks for Real-World 3D Object Understanding*, CVPR 2022; retain the current licence and indicate changes. |
| [OmniObject3D official repository](https://github.com/omniobject3d/OmniObject3D) | CC BY 4.0 | Cite Wu et al., *OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation*, CVPR 2023; comply with authorised OpenXLab/official mirror access. |

ABO's current licence supersedes older announcements describing it as noncommercial.
The [official dataset page](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
links the licence copy and requires attribution and indication of modifications.
External retail brands, people and trademarks still require care in downstream use.

The 24 campaign briefs and the controlled hypothetical assistance KB/scenarios were
authored for this evaluation with AI assistance. They are distributed as evaluation
inputs, not model results, third-party benchmark data or human feedback. Unit-test
fixtures are separately labelled synthetic and confined to temporary test paths.

## Exact selected model identities

| Selected model / official source | Acquisition record to retain |
|---|---|
| [Marqo/marqo-fashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP) | Card, code/weight licence, full revision, native open_clip configuration |
| [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) | Card, licence, full revision, native processor |
| [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | Exact `sdxl_lightning_4step_unet.safetensors`, licence and compatible SDXL base-model licence/revision |
| [black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) | Exact distilled 4B FP8 checkpoint, licence, full revision and supported FP8-loading instructions |
| [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | Card, licence, full revision and any quantisation provenance |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Card, licence, full revision, quantisation and thinking-mode configuration |
| [SPAR3D code](https://github.com/Stability-AI/stable-point-aware-3d) / [weights](https://huggingface.co/stabilityai/stable-point-aware-3d) | Separate code/weight licences, gated-access approval, checkout SHA and low-VRAM settings |
| [SF3D code](https://github.com/Stability-AI/stable-fast-3d) / [weights](https://huggingface.co/stabilityai/stable-fast-3d) | Separate code/weight licences, gated-access approval and checkout SHA |

Verified model licence details and implementation constraints are recorded in
`config/models.yaml` and the model compatibility documentation. A link above is
not a claim that weights were acquired, a licence was accepted, or inference ran.
Never bypass gated-model access or substitute a different model without approval.

## Dependencies and disclosure

The Python dependency specifications identify runtime packages. Save exact installed
versions with the execution environment and retain applicable notices when packaging.
The orchestration/metrics code is developed for this project; official model loading
patterns are attributed through the model sources above. Published benchmark numbers
are contextual literature only and must not be entered as local observations.

This repository was produced with AI-assisted programming and documentation. The
student must follow the assessment's permitted-use and disclosure rules; attribution
does not convert a prohibited use into a permitted one. Do not label AI-authored
reviews as human feedback or claim student-independent development.
