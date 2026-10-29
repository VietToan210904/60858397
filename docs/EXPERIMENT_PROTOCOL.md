# StyleSphere evaluation protocol, version 1

Completion note, 29 October 2026: all 584 original computational attempts across
four pairs/eight models are recorded. Separate supplements now cover 600 retrieval
product profiles, 184 visual outputs and all 160 shopping attempts with explicitly
AI-authored assessments. A separate real-reference GSO experiment completed four
development and 20 test reconstructions per 3D model. Human worksheets remain
blank. These additions do not retrospectively turn AI labels into independent
human validation or change the original ABO experiment's missing references.

This document describes the original design with dated completion notes. Exact
inference settings, sources and input bytes are bound by the retained deployment
freezes. The later AI and reference procedures have their own frozen protocols;
see [AI visual assessment](AI_VISUAL_ASSESSMENT_PROTOCOL.md),
[reference reconstruction](REFERENCE_RECONSTRUCTION.md) and
[reproduction commands](REPRODUCIBILITY.md). Do not tune settings on final outcomes
or rewrite frozen raw records. A changed procedure requires a separate version
with its relationship to earlier evidence declared.

## Research scope and experimental units

The user confirmed the four-pair/eight-model Evaluation Codes and Data scope and
explicitly permitted and encouraged AI use on 29 October 2026. The supplied PDF's
seven-pair/no-AI wording is historical document context, distinct from these
current instructions. AI development and assessment are disclosed; no human
feedback, external course amendment or independent validation is claimed. See
[scope confirmation](ASSIGNMENT_ALIGNMENT.md).

| Pair | Shared cases | System A | System B | Primary outcome | Unit for uncertainty |
|---|---|---|---|---|---|
| Retrieval | 600 real ABO photos; 120 controlled AI-assisted queries; human qrels blank | Marqo-FashionSigLIP | SigLIP 2 Base | Human-primary nDCG remains ungraded; separate AI/source-proxy nDCG and coverage | query; 12 query families for supplementary bootstrap |
| Generation | 24 briefs, four categories, three seeds; 72 actual outputs/model | SDXL-Lightning 4-step | FLUX.2 Klein 4B FP8 | Human usable-concept rate remains ungraded; separate AI ordinal/shortlist opinions | prompt, with its repeated images |
| Assistance | 80 hypothetical scenarios/model and shared read-only KB | Phi-4-mini-instruct | Qwen3.5-4B | Strict protocol outcomes, structured checks and separate AI semantic/grounding assessments | scenario; eight categories for supplementary sensitivity bootstrap |
| Reconstruction | Original 20 real ABO photos; separate GSO four development/20 test scans | SPAR3D | Stable Fast 3D | Original mesh/export metrics and AI previews; separate normalized/reference-assisted shape scores | source object; category-stratified test bootstrap |

The 24 generation briefs contain six examples in each of product-focused,
editorial, seasonal and multi-attribute categories. The 80 assistance scenarios
contain ten each of product details, inventory, delivery, returns, multiple business
records, unavailable information, prompt injection and unauthorised actions. These
are authored evaluation inputs. They are not sampled customer queries, observed
transactions, benchmark measurements or model responses.

## Freeze and provenance

1. Acquire authorised source assets separately after estimating bytes and checking
   free space; obtain approval before any download exceeding 5 GB.
2. Record the source repository full commit SHA or immutable object identifiers,
   licence copy, accessed date, input file hashes and any filtering/exclusions.
3. Prepare local metadata using `python -m stylesphere.cli prepare --pair PAIR`.
   Configure local imports through `datasets.PAIR` in the evaluation config.
4. When collecting human judgements, keep unknown labels blank and retain actual
   reviewer identity/date. The completed AI supplements use separate authored
   ledgers and protocol hashes; they never populate human worksheets. Retrieval
   product profiles were frozen before reviewers opened saved model rankings.
5. Validate all schemas and assets, archive the frozen manifests and configuration,
   and record the split. Any pilot cases used for settings must be separate from
   the final reported cases. The retained original deployment freezes and separate
   GSO development/test freeze identify the actual final cases. AI assessment is
   explicitly post-hoc relative to original inference.

Preparation is idempotent: existing populated catalogues, queries, object manifests,
briefs, scenarios, KB records and human labels are preserved. Changing a config
does not silently rebuild a frozen set. Archive the existing dataset folder before
creating a new version; the new manifest must identify the new source inputs.

## Retrieval and independent relevance assessment

The completed run uses the shared 600-photo ABO catalogue and 120 authored,
hypothetical queries. The following human-annotation procedure remains available
for future independent validation; no such human grades are claimed here. The
separate [AI/source supplement](../results/retrieval/ai_assessment/v1/README.md)
retains all 600 manually inspected product profiles and derives 72,000 proxy
query/product grades, including explicit unknowns. Its grades and sensitivity
scenarios are separate from the original human qrels.

Prefer an ABO fashion subset for the primary held-out retrieval exercise where
available; use Fashion200K only with its known model-evaluation exposure clearly
stated. Match source products by stable source identifiers rather than filenames.
Hash-based product IDs are `p-` plus the first 16 hexadecimal digits of
SHA-256(`source:source_id`). Query IDs may be supplied by the human author; if
omitted during import they use `q-` plus the corresponding query-text hash prefix.

The local metadata importer uses seeded round-robin category sampling, shuffled
within each category. This is independent of either compared model. Configurable
sample counts may be smaller for pilots; never describe a small pilot as 600/120.
Record source categories and exclusions rather than assuming every source contains
all requested clothing/accessory categories.

Human query authors should cover category, colour, material, style and genuinely
ambiguous requests, without copying the exact catalogue descriptions verbatim.
Predefine intended attributes without consulting either model's rankings. Keep the
query design worksheet, author identity/pseudonym and date with the frozen version.

`annotation_candidates.csv` presents query-product pairs in a deterministic random
order and contains no model ranking or inferred relevance. Default: every product
for every query (600 × 120 = 72,000 decisions at target size). An explicitly
configured random candidate pool reduces effort but also reduces judgement
completeness and can miss relevant items. Preserve the pool method and seed.
No output from either evaluated model is used to create final relevance grades.

Use the following ordinal rubric:

| Grade | Meaning |
|---|---|
| 3 | Meets the product type and all visible mandatory requested attributes |
| 2 | Useful near match; one noncritical attribute absent or uncertain |
| 1 | Weakly related alternative with a major requested attribute mismatch |
| 0 | Irrelevant or wrong product type |
| blank | Not judged; never reinterpret as an observed grade 0 |

Two assessors should independently grade at least a prespecified overlapping subset
(recommended 20%) while blinded to model identity and rank. Record disagreement and
adjudication rather than averaging ordinal labels without justification. This is a
recommended human annotation procedure; no human assessors or human agreement
scores currently exist. The separate AI batches were disjoint, so they do not
provide inter-rater reliability either.
Import completed labels using `datasets.retrieval.annotations_path`; each final row
requires `annotator_id` and `annotation_date`. Do not overwrite populated qrels
without explicitly archiving a prior version.

Set `qrels_complete=true` only after every product in the frozen catalogue was
judged for that query. Recall@K is interpretable as catalogue recall only under
that condition. With incomplete pools, report the policy for unjudged retrieved
items and sensitivity to coverage. nDCG/MRR/Hit describe performance relative to
available judgements, not exhaustive relevance. Queries without any judged relevant
products have undefined relevance-based metrics and must be reported separately.

## Generation and blinded human evaluation

Use the exact same brief for the two models and seeds 8410, 8411 and 8412. Equal
seed numbers do not imply equal latent noise across architectures. Record each
native scheduler, inference steps, guidance, resolution, checkpoint revision,
precision, offloading and processor. Retain official SDXL-Lightning four-step
Euler/trailing/zero-guidance settings and FLUX.2 Klein's supported FP8 procedure.
Report deployment configurations rather than an isolated architectural effect.

Assign random reviewer-facing IDs without model names. Store the ID-to-model map
separately; do not reveal it until a reviewer completes ratings. Ratings are blank
until entered by a person. Review each output against its mandatory criteria and
record visible failures. The actual frozen worksheet anchors are 1=unacceptable/
major defects, 2=poor/substantial revision needed, 3=adequate/some revision needed,
4=good/minor revision needed, 5=excellent/ready for the intended brief.

Score prompt adherence, visual coherence, composition and usability separately.
The additional binary usable decision asks whether the concept can be shortlisted
for further human design work without correcting a major factual/structural flaw;
it is not approval to publish an advertisement. Keep reviewer IDs, timestamps and
comments. Report reviewer disagreement and missing reviews. Similarity scores are
supplementary and do not replace human judgement. Bootstrap whole prompts, retaining
both models and all repeats of a sampled prompt together.

All 144 final images have a separate post-hoc AI assessment under these ordinal
anchors, with criterion evidence, confidence and unresolved shortlist decisions.
Model identities were masked in packets, but assessors knew the project and had
seen earlier QA. This is not an independent blinded human study. Null decisions
are retained; different conditional denominators are disclosed in
[decision coverage](../results/ai_visual_assessment/decision_coverage.json).

## Assistance and tool governance

The frozen KB includes prices/currency, categories, descriptions, sizes, colours,
variant inventory, delivery and return rules. Each scenario declares visible record
IDs, permitted tools, expected tool names, canonical required fact paths and an
expected answer/refuse/escalate outcome. Facts resolve directly into the KB, for
example `products.P001.price` or `products.P001.inventory.0.quantity`.

Only the application controller executes validated read-only tools. No model may
purchase, pay, alter accounts/stock, send messages or make external network requests.
Use identical tool schemas and maximum steps for both models. Log proposed actions,
validation, observations and final responses. Scope product/policy visibility to
the scenario. A prohibited request must not trigger a mutation or fabricated success.

Deterministic scoring can verify structured factual claims and tool schema validity.
Missing facts, free-form assertions and ambiguous semantic outcomes require human
review. An answer is not correct merely because it contains an expected keyword.
Do not count unreviewed responses as task successes. Document Phi/Qwen chat templates,
quantisation and Qwen thinking-mode settings; exclude private reasoning from the
customer answer while retaining auditable observable tool actions. A separate
[AI assistance supplement](../results/assistance/ai_assessment/v1/README.md)
now reviews all 160 actual attempts, including 73 protocol failures. It separates
semantic task delivery, factual support, retrieved grounding and operational
completion; semantic credit never rewrites a malformed-JSON execution failure.

## Reconstruction eligibility and standard views

The original 20 ABO photographs lack aligned meshes/calibrated source cameras;
their reference-accuracy fields remain unavailable. All 40 exported outputs have
separate AI preview observations, with hidden-surface correctness uncertain.
The additional GSO experiment uses licensed scanned meshes and controlled input
renders, with four development and 20 test objects (17 shoes/three backpacks).
Both models completed every development/test case. Its normalized shape scores
remove size/translation, and its additional rigid ICP scores explicitly use the
reference to fit pose. Neither measures absolute size/pose accuracy. Its exact
sampling, cameras, alignment and bootstrap rules are in the separate frozen
[reference protocol](REFERENCE_RECONSTRUCTION.md); they do not alter ABO records.

Select rigid or structured shoes, bags and accessories; avoid claiming this validates
deformable clothing. Preserve both the input and the raw generated mesh. Geometry
eligibility requires a reference mesh and documented coordinate alignment. If real
catalogue-camera poses are unavailable, render a controlled input from a reference
mesh, recording the renderer, exact mesh, camera matrix, lighting and background.
Such renders measure a controlled synthetic-view task, not real-catalogue accuracy.

`preparation_metadata` points to JSON containing input preparation, units,
normalisation, alignment evidence, and an explicit 4×4 `alignment_transform` where
applicable. `camera_info` points to JSON describing camera-to-world matrix,
intrinsics, image dimensions and reference silhouettes/renders when used. Mark
`geometry_eligible`, `silhouette_eligible` and `render_similarity_eligible` separately.
Known camera matrices alone do not prove that predicted and reference meshes align.

Sample surface points reproducibly, use a stated centre/scale normalisation and
declare F-score thresholds before inference. Keep raw alignment results distinct
from any additional alignment procedure, such as a declared rigid ICP analysis.
Never optimise geometry against the reference silently. Standard review views
should use fixed azimuths 0/90/180/270 degrees at 20-degree elevation, a documented
camera distance, diffuse lighting and a neutral background. Renderer/camera
parameters must be recorded with output views. Do not calculate view metrics when
matching camera or reference renders are missing. Do not infer absolute physical
dimensions from normalized geometry.

## Execution, measurements, failures and statistics

Run the two models sequentially on the same physical GPU. Use isolated environments
when dependencies conflict and save exact package versions. Record model load,
preprocessing, synchronized inference and end-to-end timings separately when
available, in seconds. Reset supported peak CUDA memory counters before measured
work; distinguish allocated/reserved PyTorch memory from whole-device usage.
Document warm-up and exclude it from primary case timings. Avoid concurrent GPU
jobs; retain every failed attempt, including OOM and blocked dependency/access cases.

Save per-case outputs and statuses, not only aggregates. Report failure denominators,
missing judgements and excluded geometric cases. Use paired bootstrap confidence
intervals (default seed 8410, prespecified iterations in configuration), matching
queries/scenarios/prompts/objects within a pair. Keep prompt repeats and object views
clustered. Summarise means/medians/spread and absolute differences; compute relative
differences only for meaningful nonzero baselines. Do not silently compare unmatched
success-only subsets. Completed comparisons support qualified conclusions about
these deployment configurations and declared assessment procedures. AI opinions,
reference-assisted shape scores and independent human evidence must remain
distinct; none establishes a universal model winner or customer approval.
