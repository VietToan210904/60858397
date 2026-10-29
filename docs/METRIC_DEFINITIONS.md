# Metric definitions and missingness

All metrics describe the actual recorded deployment configuration, including its
precision, preprocessing and implementation. They cannot isolate architectural
effects from quantization, model training data or runtime differences.

## Retrieval

- nDCG@10: DCG = sum((2^grade - 1)/log2(rank+1)), divided by ideal DCG at10.
- MRR: mean of reciprocal first rank with positive judged relevance.
- Hit@5: any positive judged relevant item among the first5.
- Recall@K: relevant items among firstK divided by all relevant items, available
  only when every catalogue product has an explicit judgement for that query.
- No positive judged item: nDCG, RR, Hit and Recall are missing, never zero.
- Unjudged items receive zero gain under incomplete judgements. Report judgement
  coverage and the possibility of pool bias. The small smoke set has no quality
  labels and cannot establish retrieval performance.
- Full ranked identifiers and similarity scores are retained. Stable product IDs
  resolve exact ties. Indexing includes preprocessing/encoding and index export;
  query end-to-end includes text preprocessing, encoding and CPU ranking.

## Generation

Successful-generation rate uses executed successes and failures. Preflight
blockers are reported separately. Valid outputs per processing second use verified
image count divided by measured inference seconds across successful and failed
attempts; do not average per-image inverse latency. Throughput is unavailable if
any executed attempt lacks its inference window; timed/untimed coverage is
retained rather than silently omitting the missing window. Primary usable concept rate
counts each image once: all supplied binary usability votes must agree. One
completed vote suffices; disagreement requires human adjudication and remains
missing until resolved. Supplemental usable vote rate averages individual votes.
Five-point adherence, coherence, composition and usability ratings have explicit
worksheet anchors. Average ordinal ratings within an image, images within a
prompt, then prompts within a model. Report assessment coverage and disagreements;
missing ratings remain missing. Development smoke reviews are excluded.
Human quality scores are not supplied by CLIP or other automatic similarity.

## Shopping assistance

Exact structured claims are checked against facts returned by permitted tools.
Factual correctness = supported structured claims / submitted structured claims;
unsupported-claim rate = unsupported structured claims / submitted structured
claims. With no claims these are missing. Required fact coverage and expected
outcome produce a separate structured_task_success indicator. Free-form prose
and the customer's complete task require human review: task_success remains
missing until a human adjudicates it. These structured metrics do not establish
the correctness of every statement in prose.
Structured answer-quality summaries are conditional on completed valid final
responses. Protocol failures remain in execution-failure denominators; never
interpret a conditional structured-claim rate as success across every attempted
customer task. Reports retain the eligible count beside each outcome.

Human binary task/prose outcomes require unanimous nonblank votes per scenario;
one assessor counts. Disagreement remains missing pending documented human
adjudication. Supplemental fields ending `_vote_rate` describe the mean reviewer
vote and must not be reported as adjudicated task-success rates. Preserve original
assessments and the audit trail if adjudication later resolves disagreement.

Tool schema validity is valid proposals / proposals. Tool selection appropriateness
is exact agreement with the declared expected tool set, with no invalid proposals;
this strict controlled-scenario metric is not a general semantic agent judge.
Every proposed action is validated by the application. Tool count includes rejected
proposals. No tool mutates data or performs external actions. Execution failures,
unsupported claims and unauthorised requests are distinct outcomes.

## Reconstruction

Area-weighted triangle sampling uses uniform barycentric coordinates, fixed seed
for prediction and seed+1 for reference. The primary Chamfer Distance is
0.5*(mean_a min_b ||a-b||^2 + mean_b min_a ||b-a||^2); chamfer_l1 records the
unsquared alternative. F-score is 2PR/(P+R) for nearest-neighbour distances within
predeclared thresholds; P+R=0 gives0. Common normalization, when selected, uses
the reference bounding-box centre and diagonal for both already-aligned meshes.
It never fits each prediction independently. Reference-dependent scores require
predeclared affine alignment and known coordinate units. No hidden ICP is used.

Silhouette IoU needs matching cameras; two empty masks yield missing. View
similarity is normalized RGB MAE and PSNR on matching reference renders; exact
matches have PSNR missing instead of JSON Infinity. Related views aggregate per
object. Mesh validity checks finite nondegenerate triangle geometry; watertightness
is separate and does not imply correct topology or physical dimensions.
Watertightness uses the exported indexed topology without welding vertices.
Texture UV seams can split otherwise coincident vertices, so a false value alone
does not establish missing geometry.

## Timing, memory and statistical analysis

Execution failure rate is failed executions divided by successful plus failed
executions. Preflight blockers and unexecuted planned cases are separate counts,
never model failures. The complementary execution success rate uses the same
denominator. All logged attempts count, so retries cannot erase prior failures;
the raw failure table also retains every failed attempt. Paired reliability uses
the first executed attempt per matched case. Successful output comparisons are
conditional on retained successes and report matched/eligible counts separately.
Failed traces and available artifact hashes remain verifiable.

Synchronized wall-clock seconds measure actual windows, with model load, input
preprocessing, inference and export separated where implemented. Warmup outputs
are discarded. CUDA peak allocated/reserved values are MiB (2^20 bytes), include
allocations for model parameters and differ from total VRAM in nvidia-smi. These
allocator counters do not establish physical residency. No estimated
VRAM or timings are benchmark measurements.

For reconstruction, `preprocessing_s` covers case seeding, image reading and
validation, model-native foreground framing, and saving the prepared PNG.
`inference_s` measures the synchronized `run_image` call, including its geometry
and texture work; `export_s` measures GLB export separately. `end_to_end_s` is the
measured case wall interval through export or failure, with the measured warmup
interval subtracted. It excludes model loading and the later parent-process mesh
validation and preview rendering. `warmup_excluded_s` is diagnostic and appears
only when a warmup ran. Both models use CUDA BF16 autocast; SPAR uses its official
low-VRAM mode. Seeds are reset again immediately after warmup.

BF16 autocast is the verified execution setting, not a measured complete
parameter-dtype distribution. SPAR's low-VRAM implementation performs deferred
module loads/swaps inside `run_image`; these belong to inference/worker E2E even
when its separately timed constructor is inexpensive. Constructor timing must not
be interpreted as loading every module into resident GPU memory. The frozen
`remesh="none"` condition uses neither optional triangle nor quad remeshing.

Mean, median and sample standard deviation retain eligible counts. Bootstrap CIs
resample matched independent cases using seed 8410 and2000 iterations by default.
Prompt-level averages and object-level aggregates prevent repeated images/views
from falsely increasing independent sample size. A single independent cluster has
no CI. Pairwise comparisons require shared protocol, input and case identifiers.
Relative differences are missing when the reference mean is zero. Effect sizes,
coverage, practical usefulness and deployment asymmetries matter beyond CI overlap.
