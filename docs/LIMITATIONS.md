# Limitations and responsible interpretation

All 584 original planned computational attempts were recorded, including 73 strict
shopping-protocol failures. Separate completed supplements contain AI assessments
of 600 retrieval product photos, 184 visual outputs and 160 shopping attempts.
The separate GSO reference experiment completed four development and 20 test
objects per 3D model. See `IMPLEMENTATION_STATUS.md`, `environment/` and the
[final synthesis](../results/final_evaluation/REPORT.md) for retained evidence.
These completed analyses have specific limits: no independent human feedback is
present, and normalized/reference-assisted scores do not establish absolute
physical accuracy or commercial readiness. Original ABO reference metrics remain
unavailable; the new GSO dataset does not retroactively supply them.

## Assessment scope and AI assistance

The user confirmed four pairs/eight models for the Evaluation Codes and Data
component and explicitly permitted AI use on 29 October 2026. AI assistance is
disclosed. The PDF's different seven-pair/no-AI wording is preserved as historical
context rather than treated as a current blocker. No external course amendment
or independent student authorship is claimed.

## Data and external validity

Fashion200K is a known evaluation dataset for Marqo-FashionSigLIP according to the
[official dataset card](https://huggingface.co/datasets/Marqo/fashion200k).
Its reuse cannot establish unseen-domain generalisation. Different source, brand,
photographic and category distributions also limit any ABO-derived conclusion.
Category-balanced small subsets do not reproduce the deployment catalogue's natural
class frequencies. Captions and human query construction may favour particular
attributes; retain their provenance and analyse categories separately.

Human graded relevance can be subjective and incomplete. Random independent pools
avoid using either model's predicted rankings to create final labels, but small pools
can miss relevant items. Do not describe incomplete-pool Recall@K as exhaustive
catalogue recall. Unjudged items and queries without relevant judgements need explicit
handling. No actual human assessments are present merely because a worksheet exists.

The completed retrieval supplement uses AI-authored product types and visual
traits plus exact multilingual source citations, frozen before reviewers opened
rankings. It is not independent human ground truth. Its 220 unknown proxy grades
stay null; main metrics use an explicit zero-gain scenario and additional
sensitivity/bound analyses. Fifty of 120 queries have complete catalogue proxy
labels. Catalogue recall must not be inferred from the incomplete queries.
Product batches were disjoint, so assessor differences and inter-rater reliability
are unmeasured. The 120 queries are controlled authored requests, not customer
samples; related variants are clustered into 12 query families.

The 24 promotional briefs are a small authored set. Images from the same brief are
correlated; seed equality does not equal latent-noise equality. Blinded human ratings
remain subjective and are required for primary quality/usability outcomes. Photorealism
or text-image similarity does not prove usable advertising, valid anatomy, product
accuracy, intellectual-property clearance or truthful representation of real stock.

All 144 generation outputs and 40 original ABO reconstructions received
supplementary AI visual annotations. Assessors helped implement the project and
had seen earlier QA; masked model labels do not make this an independent blinded
study. Shortlist opinions include 13 uncertain FLUX and one uncertain SDXL image.
Conditional prompt means use 23 versus 24 prompts, so raw decisions and complete
matched-pair coverage must accompany comparisons. Human primary outcomes remain
ungraded. Confidence labels are self-reported certainty, not probabilities.

The 80 assistance scenarios and 12-product business KB are hypothetical controlled
inputs. They support reproducible tests of tool use and factual grounding, not claims
about live customer satisfaction, call-centre workload, policy legality or real sales.
Reference-fact matching cannot reliably grade all unrestricted natural language.
Missing or ambiguous claims need transparent human review; unreviewed outcomes are
not automatically correct. Prompt-injection examples cover selected attack patterns,
not an exhaustive security assessment.

AI assistance review covers all 160 attempts, including the 73 original protocol
failures. Semantic credit for readable prose does not make invalid controller
output operationally successful. Factual support, retrieved grounding and task
delivery are different axes. Their different missing/not-applicable denominators
are retained. The implementation-contributor assessor is not a customer or an
independent expert, and its reliability is unmeasured.

3D evaluation is conditional on reference integrity and alignment. Normalized shape
metrics do not measure absolute physical size. Real catalogue images often lack
calibrated cameras; matching-camera metrics cannot be inferred from plausible
renders. Controlled reference-mesh renders isolate a reproducible image-to-geometry
task but introduce a synthetic-to-real gap. Rigid accessories do not validate loose
garments, deformable fabrics, unseen surfaces or physical product fit.

The GSO test set contains 17 shoes and three backpacks selected lexically from
source metadata, with four disjoint development objects. It is small, category-
imbalanced and ASICS-heavy; training-set overlap is unknown. Fixed input cameras
can show a shoe's heel/end rather than its semantic front. Independently centering
and scaling predictions/references to unit bounding-box diagonal removes size and
translation and depends on orientation. ICP explicitly uses reference geometry
to fit pose. Its measurement samples are separate from registration samples,
but that separation does not make alignment reference-free. F-score, Chamfer and
supplementary silhouette IoU can favour different outputs; report all of them.
Controlled renders do not establish accuracy on natural catalogue photographs.

## Hardware, deployment and inference

The target machine is the user's RTX 4070 SUPER with 12 GB VRAM. Only the genuine
doctor report establishes hardware visible to this execution session. A CPU test or
library import is not a GPU model smoke test. Gated weights, network failures,
missing CUDA extensions and native Windows incompatibilities are legitimate blockers,
not permission to report estimated inference outcomes or silently use hosted services.

Quantisation, FP8/BF16/FP16 precision, CPU offloading, architecture-specific processors,
thinking modes and runtime versions can all affect accuracy, latency and memory.
The study compares actual deployment configurations, not an isolated causal effect
of model architecture. Cold loading, warm inference, end-to-end latency and device
memory counters must be named separately. Failures and skipped cases affect the
denominator and must be reported, with matched-case comparisons clearly identified.

## Privacy, fairness and governance

The distributed hypothetical inputs contain no real customers or personal data.
Do not add private conversations, addresses, payment details, access tokens or
account identifiers to traces or submission archives. Read-only tools reduce the
action surface but do not guarantee that all generated text is safe or correct.
Do not allow the model to execute arbitrary function names, code or external URLs.

Editorial prompts include age, skin tone, body-size and disability variation, but
24 briefs cannot measure demographic fairness. Product retrieval may inherit image,
gender presentation, body representation, language and regional biases. Avoid
unmeasured claims of equitable outcomes. A human must verify misleading content,
brand likenesses, texture/material claims and advertising suitability.

Dataset/model licences, gated access and asset rights need revision-specific review.
A repository licence does not settle every photograph, trademark or person-related
right. Redistributable flags default false; references and provenance do not grant
permission. No weights or large third-party archives should be packaged by default.

## Statistical and business conclusions

Paired samples, fixed seeds and clustered resampling improve repeatability but do
not remove annotation bias, small-sample uncertainty or multiple-comparison risk.
Primary metrics and thresholds must be frozen before outcome inspection. Do not
interpret a small mean advantage as meaningful without uncertainty, failure rates,
review evidence and an appropriate business threshold. No deployment, autonomous
customer action or publication recommendation is supported until the corresponding
actual evidence and human review exist.
