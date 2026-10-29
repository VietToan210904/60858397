# Protocol amendments and actual sampling frame

Recorded 29 October 2026, Australia/Sydney. The original protocol states target
sample sizes and intended accuracy outcomes. Actual data availability changes the
claims that can be supported; it does not justify inventing labels or references.

| Item | Original target | Actual frozen inputs and interpretation |
|---|---|---|
| Retrieval | 600 images, 120 human-authored shopping queries | 600 distinct licensed ABO product images from `listings_0`, selected with seed8410, and120hypothetical AI-assisted controlled queries written before model outputs. These are controlled requests, with no human author or customer sampling claim. No relevance grades exist. |
| Reconstruction | About24objects with suitable geometric references |20real ABO fashion product photographs:5boots,5shoes,10handbags. A predefined foreground-mask rule excluded1of21candidates. There are no matching reference meshes or calibrated cameras, so Chamfer/F-score/silhouette/reference-render accuracy are unavailable. |
| Generation development | Separate pilot before final24briefs | One additional loafer brief (`dev-gen-001`) is excluded from all final24briefs. Deployment was frozen before the72SDXLbenchmark outputs; the development output is never counted as a benchmark case or human rating. |
| Assistance development | Separate pilot before final80scenarios | Two additional price/inventory scenarios in `datasets/assistance_smoke` are excluded from the final80. Freeze deployment settings after development checks and before final inference. |

The retrieval sampling frame excludes apparel and covers one ABO metadata shard,
rather than the complete ABO corpus. The candidate annotation pool contains30
products per query, selected independently of model scores. It is incomplete and
may miss relevant products. The ungraded worksheet is not a labelled benchmark.

The original target remains visible in `EXPERIMENT_PROTOCOL.md`. This amendment
was written after the240retrieval query executions and the SDXL development
smoke, while Phi/Qwen/FLUX/SPAR3D/SF3D main-model inference had not run. Retrieval
inputs and query wording were already fixed before those retrieval outputs;
selection was not changed in response to rankings. No quality labels or scores
were examined to choose these amendments. Subsequent execution freezes and exact
input hashes are retained in `environment/` and per-case records.

Reconstruction inference can support mesh validity, export reliability, resource
use and qualitative inspection. It cannot establish reconstruction accuracy
against unseen true geometry. Missing quality evidence remains explicit in reports.

## Assistance development adjustment

The first Phi development run fit in BF16 and produced one completed response and
one strict JSON parsing failure. The completed response omitted the `products.`
prefix from its structured claim paths. Both original traces remain unchanged.
Before either model sees the final80scenarios, controller protocol version2adds
the exact canonical paths of facts already returned by permitted tools to the
read-only observation. This exposes no expected answers or unreturned KB facts.
Both models receive the same format, output budget and strict parser. No automatic
JSON repair or grading of prose is added. Version1replay remains available for
the original diagnostic traces; final deployment requires fresh development
checks and a recorded freeze under version2.
