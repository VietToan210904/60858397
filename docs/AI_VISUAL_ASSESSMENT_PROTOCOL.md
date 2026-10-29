# Post-hoc supplementary AI visual assessment

This protocol is fixed before the new annotations, after all original model
outputs were generated and some examples were already inspected for QA. It is
therefore **post-hoc supplementary AI assessment**. Model labels and source paths
are masked in review packets, but assessors know the project models and earlier
QA exists. This is not independent human feedback or a blinded human study.
No model parameters, original benchmark records or human worksheets change.
The current assistant inspects actual images through its image-viewing tool;
there is no separate judge API call or new GPU model inference. Record the actual
assessor identity and timestamp. If the serving model/version is unavailable,
write `not_exposed`; do not invent a model version or a judge seed.

The exported `protocol.json` binds this document, configuration and exporter
source SHA256. Its `protocol_sha256` and selection manifest must exist before
the first annotation. Never revise this protocol to improve observed outcomes.
Any future revision gets a separate directory and declares its relationship to
these annotations. Assessment JSONL shards are separate from every human CSV.

## What to inspect and record

Generation packets contain all 144 final images in twelve shuffled batches of
twelve, with opaque IDs and exact frozen briefs/mandatory criteria. Each image
is shown at 512 by 512 within a labelled contact sheet. Individual display PNGs
can be opened separately. Fine lettering or small details that cannot be resolved
at that display resolution remain uncertain. Do not guess material composition
or hidden anatomy. Note the actual visible region supporting each observation.
Do not change the image to remove defects. Metadata is removed only in new copies.

For each frozen mandatory criterion record `met`, `absent` (contradicted or
missing), or `uncertain`, plus visible evidence. Use the exact human worksheet's
published scale labels for the separate AI ordinal scores: 1 unacceptable/major
defects; 2 poor/substantial revision; 3 adequate/some revision; 4 good/minor
revision; 5 excellent/ready for the intended brief. These ordinal annotations
are AI judgements, not measurements of commercial value. Score AI prompt
adherence, visual coherence, composition and usability independently.

`ai_usable_concept` is a supplementary AI shortlist opinion: 1 only when every
mandatory criterion is observed met and no major structural/brief failure needs
correction; 0 requires an observed major failure; otherwise null. It is not the
primary human `usable_concept_rate`, a human vote, or approval to publish.
Confidence is self-reported low/medium/high certainty, not a probability. Give
uncertainty reasons, especially for ambiguous fine details. Abstention keeps all
scores and the usable-concept decision null and explains why.

Reconstruction packets contain the source image and the four existing previews
for each of 40 model/object outputs. One object is one assessment. Score visible
input fidelity, shape coherence, texture coherence and cross-view consistency.
The four criterion keys are defined in the exporter. Unseen-surface correctness
must remain `uncertain`, because these objects have no aligned reference mesh.
The reconstruction `ai_usable_concept` field stays null. The previews have
per-object display centering/scale, and their azimuths are not calibrated catalogue
views. A visual impression cannot override the independently stored mesh-validity
flag, establish watertightness, or prove that hidden surfaces are correct.

Record actual observations with an asset name, description and optional normalized
rectangle `[left,top,right,bottom]`. Avoid attributing model identity or inferring
the generating algorithm. Use the task's existing error taxonomy; `none_observed`
means inspected with no visible error, never an uninspected output. A single
primary annotation is permitted per output; repeated AI runs are correlated and
must not be presented as independent raters or fabricated consensus.

## Aggregation and reporting

Validate schema, timestamps, image hashes, criterion coverage and explicit AI
identity before aggregation. Preserve every observation, uncertainty and missing
value. Generation averages scored seed images within each prompt, then gives
prompts equal weight within a model. Report image and prompt coverage. Pair model
outputs only on matching prompt/seed/input fingerprints, and bootstrap whole
prompts. Reconstruction aggregates at object level, then model, and bootstraps
matched objects. Related preview views are not independent observations.
Ordinal means are descriptive summaries. Confidence intervals describe sampling
variation among these prompts/objects, not uncertainty in the AI judge, human
preferences, real customers or general deployment performance. Keep these results
in an explicitly labelled AI supplement, alongside the still-blank human review.

## A future controlled-reference reconstruction extension

Use a new disjoint dataset/protocol/output directory. The pinned SF3D and SPAR3D
code both export `R_y(+90 degrees) R_x(-90 degrees)`, mapping native coordinates
`(x,y,z)` to `(-y,z,-x)`. Their native conditioning cameras sit on +X, so the
exported conditioning-facing side is -Z. They assume different focal geometry
(SF3D: 40-degree vertical field of view, distance 1.6; SPAR3D: 0.591627 radians,
distance 2.2) and different foreground framing. This known rotation does not
establish predicted scale, translation, physical dimensions or pose accuracy.

Acquire licensed textured reference scans, reserve development objects before
testing, retain exact source identities, and render one common controlled input
per object plus held-out reference views. Record all source-to-canonical matrices,
units, cameras, crop/resize image transforms and lighting. Test an asymmetric
axis-marked synthetic fixture to verify handedness, export rotation and camera
projection without running or scoring an AI model. Use disjoint development
objects to validate any fixed model-global similarity transform and freeze it
before testing. If a global transform is insufficient, keep primary alignment-
dependent metrics unavailable; any explicitly declared per-object alignment is
a separate shape-after-alignment analysis, never absolute metric reconstruction.
Do not reuse the existing independently centered/scaled preview meshes for GT
Chamfer/F-score or camera-matched silhouette comparisons.
