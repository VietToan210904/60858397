# Illustrative AI visual findings

These are post-hoc supplementary AI observations of genuine saved outputs. They
are separate from human feedback, physical product inspection and commercial
approval. All 144 generation human-rating rows remain blank. Assessors viewed
masked packets before unmasking, but knew the project models and had seen earlier
QA examples; this was not an independent blinded human study. The frozen
[assessment protocol](AI_VISUAL_ASSESSMENT_PROTOCOL.md),
[184-output summary](../results/ai_visual_assessment/summary.json) and
[decision coverage](../results/ai_visual_assessment/decision_coverage.json) retain
the full population, including uncertainty. No model settings, original raw
records or existing assessments were changed for this document.

## Example selection

Selection was mechanical **after** assessment, for illustration rather than an
estimate of prevalence. Generation pairs are sorted by `(case_id, seed)`. Select
the first resolved model disagreement, then the first different prompt with two
positive AI shortlist decisions, then the first different prompt with any null
decision. Show both models for each selected pair: six images in total.

For reconstruction, sum all four AI ordinal scores across both models for each
object. Select the lowest and highest totals, resolving ties by ascending object
ID, and show both models: four outputs. This deliberately shows extremes of this
AI assessment, rather than a representative sample. All examples below were
reopened and visually checked while writing this document. The
[documentation audit](../environment/ai-visual-documentation-audit.json) retains
the selected IDs, source paths and hashes.

## Generation: six images

| Case / seed | Model and assessment ID | Visible evidence and limits |
|---|---|---|
| `gen-001` / `8410` | FLUX.2 Klein FP8, `ai_a583eb93f37f6ef1` | One burgundy bag has a short handle and gold-coloured front clasp on an ivory plinth, with open space above and no visible lettering. Frozen AI shortlist decision: **1**. [512-pixel assessment image](../results/ai_visual_assessment/images/ai_a583eb93f37f6ef1_image.png); [original output](../results/generation/9a379c1a438b447a871f4c16f5e78417/flux2_klein/gen-001_8410.png). |
| `gen-001` / `8410` | SDXL-Lightning, `ai_6ea2bb07b3134315` | The burgundy bag is coherent, but small gold lettering is visible low on its front, contradicting the no-lettering criterion. Frozen AI shortlist decision: **0**. Brass-coloured hardware is visible; its exact clasp function is ambiguous in this view, so this example does not independently establish that functional detail. [Assessment image](../results/ai_visual_assessment/images/ai_6ea2bb07b3134315_image.png); [original output](../results/generation/c72db9e579f046999779f6fba62b7849/sdxl_lightning/gen-001_8410.png). |
| `gen-002` / `8410` | FLUX.2 Klein FP8, `ai_a823c76295567c9f` | Two white low-top trainers have green heel areas and off-white soles on grey upholstery. No logo or words are identifiable at the assessment resolution. Frozen AI shortlist decision: **1**. This is an appearance judgement, with no footwear performance inference. [Assessment image](../results/ai_visual_assessment/images/ai_a823c76295567c9f_image.png); [original output](../results/generation/9a379c1a438b447a871f4c16f5e78417/flux2_klein/gen-002_8410.png). |
| `gen-002` / `8410` | SDXL-Lightning, `ai_3b97e29220fd2a2f` | Two white low-top trainers have green heel panels, off-white soles and additional green laces. The extra lace colour does not contradict the frozen mandatory criteria. Frozen AI shortlist decision: **1**. Fine tongue marks cannot establish absence of every possible brand-like detail beyond the displayed resolution. [Assessment image](../results/ai_visual_assessment/images/ai_3b97e29220fd2a2f_image.png); [original output](../results/generation/c72db9e579f046999779f6fba62b7849/sdxl_lightning/gen-002_8410.png). |
| `gen-004` / `8411` | FLUX.2 Klein FP8, `ai_e124ab7d880f204b` | A camel cable-knit scarf and folded matching pieces sit on a light wood tabletop. Fingers and thumbs are not exposed clearly enough to verify the requested gloves. Frozen AI shortlist decision: **null**, preserved as unresolved rather than scored as success or failure. [Assessment image](../results/ai_visual_assessment/images/ai_e124ab7d880f204b_image.png); [original output](../results/generation/9a379c1a438b447a871f4c16f5e78417/flux2_klein/gen-004_8411.png). |
| `gen-004` / `8411` | SDXL-Lightning, `ai_34853c68d6148131` | A scarf with ribbed/fringed ends and a hat-like knit object are visible. No matching glove pair is identifiable. Frozen AI shortlist decision: **0**. The defect concerns the requested product set, rather than the overall attractiveness of the knit rendering. [Assessment image](../results/ai_visual_assessment/images/ai_34853c68d6148131_image.png); [original output](../results/generation/c72db9e579f046999779f6fba62b7849/sdxl_lightning/gen-004_8411.png). |

The images illustrate why visual coherence and brief compliance are separate
scores: a plausible product photograph can still add lettering or omit a required
item. Shared seeds identify paired cases; they do not imply identical latent noise
or equivalent model randomness.

## Reconstruction: four outputs

All four selected outputs use seed `8410`. Each linked card contains the source
photograph and all four saved previews. These are the original ABO experiment's
uncalibrated, independently centered/scaled display views, not the separate GSO
controlled-reference experiment. Apparent brightness or gloss can reflect
rendering and material settings. Hidden-surface correctness stays uncertain.

| Case | Model and assessment ID | Visible evidence and limits |
|---|---|---|
| `abo-B07WRJFN95` | SF3D, `ai_58f365408e47307a` | The wallet front remains recognizable, but side views show a vertical panel with a broad projecting base, resembling an L-shaped slab. The AI shape-coherence score is **2/5**. [Source plus four views](../results/ai_visual_assessment/images/ai_58f365408e47307a_card.png); [exported GLB](../results/reconstruction/7685fb6f464d4360a1ebadb8b1994d49/sf3d/abo-B07WRJFN95/mesh.glb). |
| `abo-B07WRJFN95` | SPAR3D, `ai_9e681f89ba143dac` | Broad views show a jagged, cut-away upper region; side views are extremely thin. The AI shape-coherence score is **1/5**. This is a visible structural problem despite successful GLB export. [Source plus four views](../results/ai_visual_assessment/images/ai_9e681f89ba143dac_card.png); [exported GLB](../results/reconstruction/d088458f86e447f58e6c81f257cbb027/spar3d/abo-B07WRJFN95/mesh.glb). |
| `abo-B07R79WR4V` | SF3D, `ai_f55a4411c4379f1c` | The orange lace-up high-top, collar and thick sole remain recognizable across views. Laces and surface detail are coarse and unevenly shiny. The AI shape-coherence score is **4/5**, a relatively strong example within this small assessment. [Source plus four views](../results/ai_visual_assessment/images/ai_f55a4411c4379f1c_card.png); [exported GLB](../results/reconstruction/7685fb6f464d4360a1ebadb8b1994d49/sf3d/abo-B07R79WR4V/mesh.glb). |
| `abo-B07R79WR4V` | SPAR3D, `ai_5ed93dc6e159b7b2` | The same high-top's major parts remain recognizable, with softened detail, lighter colour and a narrower form in end views. The AI shape-coherence score is **3/5**. No absolute dimension or unseen-surface accuracy follows from these previews. [Source plus four views](../results/ai_visual_assessment/images/ai_5ed93dc6e159b7b2_card.png); [exported GLB](../results/reconstruction/d088458f86e447f58e6c81f257cbb027/spar3d/abo-B07R79WR4V/mesh.glb). |

Export success, deterministic mesh validity and AI visual opinions answer
different questions. Across all original ABO outputs, both models exported
20/20 GLBs; the strict stored mesh-valid flag is 19/20 for SF3D and 20/20 for
SPAR3D. The SF3D exception, `abo-B07XPHD4LK`, contains one degenerate face under
the zero-degenerate-face rule and is retained unchanged. Neither selected
qualitative extreme replaces that independent audit, or establishes reference
geometry accuracy. See the [40-output validation receipt](../environment/reconstruction-benchmark-validation.json).

## Denominators and interpretation

All **184/184** selected outputs received an AI annotation: 144 generation images
and 40 reconstruction outputs. Annotation coverage is not the same as a resolved
binary opinion. Generation shortlist decisions are **40 positive, 19 negative,
13 uncertain** for FLUX and **19 positive, 52 negative, one uncertain** for SDXL,
out of 72 images each. Human shortlist outcomes remain ungraded.

The frozen conditional prompt means, 0.6087 for FLUX and 0.2708 for SDXL, average
decided images within each prompt and then prompts with at least one decision:
**23 and 24 prompts**, respectively. They must not be described as 40/72 and
19/72, or as estimates with equal missingness. Conditional image rates are
40/59 and 19/71 and answer another descriptive question. There are **58/72**
complete matched seed-image decisions across 23 prompts; 14 pairs remain
unresolved. The additional complete-prompt diagnostic uses **16 prompts / 48
pairs**, with prompt means 0.6875 and 0.3333. It is a restricted coverage check,
not a replacement population or a new primary outcome.

Ordinal generation scores cover all 24 prompts per model. Reconstruction scores
cover 20 objects per model; four views do not create 80 independent observations.
Bootstrap intervals describe variation among these selected prompts or objects,
conditional on the AI judgements. They do not quantify AI assessor bias, human
preferences, customer satisfaction or suitability for publication. Multi-label
error counts can overlap and must not be added as distinct failing-output counts.
