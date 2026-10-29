# Dataset provenance and preparation

Prepared on 29 October 2026. Benchmark preparation itself downloads no third-party
assets. A separate opt-in acquisition fetched eight genuine images for smoke inference.
Subsequent approved acquisition populated 600 ABO retrieval images and 20 real ABO
reconstruction inputs. Retrieval relevance and 3D reference geometry remain absent.
The generation and assistance dataset files contain original, explicitly hypothetical
business inputs. Generated outputs are stored separately in results; no human
ratings have been inferred or filled automatically.

| Input | Current contents | Remaining requirement |
|---|---|---|
| Retrieval catalogue, queries, qrels | 600 ABO images, 120 controlled hypothetical queries, empty human qrels | Independent human relevance judgements |
| Retrieval annotation candidates | 3,600 seeded model-independent pairs | Humans supply grades, identity/date and notes |
| Generation prompts | 24 authored briefs, 4 groups × 6, 3 seeds;144 outputs recorded | Independent blinded human ratings |
| Assistance KB/scenarios | 12 hypothetical products and80 controlled scenarios;160 attempts recorded | Independent task/prose review; protocol failures retained |
| Reconstruction objects | 20 licensed ABO photos with prepared RGBA;40 model exports recorded; no reference mesh/cameras | Independent visual review; eligible reference geometry/cameras remain unavailable |
| Separate `retrieval_smoke` | 8 real Fashion200K JPEGs, 4 predefined queries, no qrels | Inference validation only; not a quality benchmark |

Each prepared folder has a `manifest.json`: input filenames, byte sizes and SHA-256,
preparation seed, explicit input kind and `model_inference_performed=false`.
Preparation timestamps are provenance, not experiment runtime. Edited human input
files are preserved; re-preparation updates the manifest to hash their current
contents. Archive/freeze an immutable version before actual inference.

## Acquiring source data separately

The [Marqo Fashion200K card](https://huggingface.co/datasets/Marqo/fashion200k/blob/main/README.md)
reports 201,624 records, about 3.467 GB compressed and 4.236 GB decoded. The original
images came from retailer aggregation, and the mirror explicitly disclaims ownership.
The Marqo card also states that Fashion200K was used to evaluate FashionSigLIP, so
it is unsuitable as independent evidence of a previously unseen distribution.
Select local subsets and retain that exposure caveat in any report.

The [official ABO download page](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
lists approximately 83 MB for listings, 3 GB for small images, 110 GB for original
images, 40 GB for spins and 154 GB for 3D models. Use listings/image metadata to
identify individual objects and acquire only selected assets. A complete archive
is unnecessary. Estimate selected files and staging/uncompressed space before
acquisition. Dataset size is separate from model weight/cache requirements.

The [official OmniObject3D repository](https://github.com/omniobject3d/OmniObject3D)
describes account-based OpenXLab access, partial-path downloads, original scans,
rendered images/cameras and processed video poses. Its full compressed dataset is
about 1.2 TB. Follow current authorised access and acquire only eligible objects;
do not copy credentials from documentation into this project or submission.

This project makes no automatic network call in `prepare`. An attempted small
metadata request through PowerShell failed with `Authentication failed, see inner
exception`; Python `requests` subsequently succeeded. The opt-in
`scripts/acquire_retrieval_smoke.py` acquired eight source JPEGs and a dataset card,
with a hard 100 MB total transfer cap. Initial response bytes totalled 169,731;
after adding the pinned upstream licence record the total is 186,032 bytes.
The source API identified full dataset revision
`b9d0e6c9c14337620e442c93f40f7906942fe893`; each cached image URL was checked against
that revision. `datasets/retrieval_smoke/manifest.json` retains image hashes, byte
sizes, source row indices, selection method and source-card hash. Signed URL query
parameters are not retained.

Re-running the acquisition script verifies existing image/source-card/licence hashes
and returns `verified_existing` without re-downloading those files. The smoke folder
retains the upstream Apache licence, its immutable source commit and SHA-256 in
`licence_record`; underlying-image rights remain a separate caveat.

The eight fixed source row offsets were declared before acquisition; four smoke
queries were predefined as “a green dress”, “a dark jacket”, “a pair of trousers”
and “a patterned blouse”. There are no relevance grades. The smoke manifest marks
`metrics_eligible=false`; successful rankings from this set cannot be described as
retrieval quality results. The final `datasets/retrieval` folder is separately
populated with 600 ABO products. Its sampling frame is only the 9,232 records in
the official `listings_0.json.gz` shard, balanced across 13 available fashion types;
it has no apparel coverage. `environment/abo-retrieval-acquisition.json` pins the
two metadata SHA-256 values and each selected row records its source image/hash.
The 120 controlled queries and 3,600 candidate pairs contain no human grades.
OmniObject3D assets have not been acquired.

For Hugging Face sources record the full 40-character revision and exact downloaded
files. For ABO record the accessed date, S3 key/version or ETag when available and
local SHA-256. ETag alone is not a reliable content SHA-256. For OmniObject3D retain
the selected archive identifiers, access date, licence copy and local SHA-256.
Archives/assets are excluded from the source submission by default.

## Local retrieval import

Supply CSV, JSON list, JSONL or local Parquet metadata. Parquet requires optional
`pandas` and a compatible Parquet engine. Accepted normalized fields are `source_id`
(or `product_id`), `image_path`, `category`, `attributes`. Fashion200K aliases
`item_ID`, `text`, `category1` and a local `image.path` are supported. Embedded
image bytes must first be exported to local image files; the importer never invents
an image or fetches remote URLs. Retain the export script/version and source hashes.

ABO listing JSONL aliases `item_id`, `product_type`, `item_name` and `main_image_id`
are supported. Provide `images_metadata_path` with CSV columns `image_id,path`
to map image identifiers to locally acquired image files. Nested multilingual
`[{value: ...}]` text is flattened for human-readable attributes. Explicitly choose
categories; inspect actual source category values before filtering.

Configure these keys under `datasets.retrieval` in a copied evaluation config:

```yaml
datasets:
  retrieval:
    metadata_path: data_private/retrieval/source.jsonl
    image_root: data_private/retrieval/images
    source: amazon-berkeley-objects
    source_revision: 'accessed-YYYY-MM-DD:YOUR-IMMUTABLE-OBJECT-RECORD'
    licence: CC-BY-4.0
    catalogue_size: 600
    query_count: 120
    categories: []  # Set inspected exact category values; [] means no filtering.
    redistributable: false
    # images_metadata_path: data_private/retrieval/images.csv
    # queries_path: data_private/retrieval/human_queries.csv
    # annotations_path: data_private/retrieval/human_annotations.csv
```

Run `python -m stylesphere.cli prepare --pair retrieval --config PATH_TO_CONFIG`.
The importer checks assets, hashes images and generates stable IDs. It selects a
seeded category-balanced subset. Files reference local assets; they do not contain
redistributable copies. Set redistribution eligibility only after checking rights,
attribution and institutional submission requirements.

Human query CSV columns are `query_id,query_text,query_category,qrels_complete`.
Set completeness false until every product is judged. Query IDs can be omitted
during import and will be deterministically generated from query text. No query
is automatically copied from the source product caption. Human annotations require
`query_id,product_id,relevance,annotator_id,annotation_date`. Grades are 0..3; blank
is unjudged. Prepare annotation candidates before either model is run and follow
the rubric and independent assessment process in `EXPERIMENT_PROTOCOL.md`.

## Local reconstruction import

Supply local `datasets.reconstruction.metadata_path` in the same supported formats.
Rows use the column names in `datasets/reconstruction/objects.csv`. Configure
`object_count` (default 24). Every object requires source image, category, source
dataset, licence and source revision. Optional paths reference a reference mesh,
camera JSON and preparation JSON. Paths are resolved relative to the project root.
Image and mesh SHA-256 values are computed when imported. Duplicate object IDs and
missing files fail preparation.

All three eligibility flags default false. `geometry_eligible=true` requires a
reference mesh and preparation metadata explaining alignment. View metric flags
also require camera metadata. These checks establish presence; the evaluator must
still validate actual mesh integrity, camera conventions and alignment suitability.
Record background removal, alpha handling, input resizing and controlled rendering
in preparation metadata. Never place predictions in reference asset paths.

The subsequently prepared final set contains 20 real ABO inputs: 5 boots, 5 shoes
and 10 handbags, selected from the frozen 600-product catalogue without consulting
3D model outputs. `selection_plan.json` freezes candidate order/seed/thresholds;
`quality_ledger.json` records 21 screened candidates and one disconnected-alpha
exclusion. Original image acquisition transferred 9,068,932 bytes. CPU rembg2.0.57 /
U2NetP produced the common RGBA inputs. Independent human quality ratings are pending.
The saved CC BY4.0 licence and `RIGHTS_AND_ATTRIBUTION.json` cover explicit
redistribution of the selected S3 photos and documented resized/matted derivatives.
All reference-geometry and matched-view eligibility flags remain false. Native
builds, reproducible preparation and missing-reference limits are documented in
`NATIVE_3D_SETUP.md`. The smoke runner uses only the first boot row from its
two-object development manifest; that boot is absent from the final 20.
The smoke set also has an explicit `RIGHTS_AND_ATTRIBUTION.json`: both original
photos were matched by hash to the official ABO S3 archive under CC BY 4.0.
It covers their RGBA derivatives and the model-specific `prepared_input.png`
crop/resize copies retained under results. This rights update changed no image
bytes or case selection; prior conservative manifests are archived in environment.

## Hypothetical business inputs

Generation and assistance inputs are original authored material created for this
project with AI development assistance. All products, prices, quantities and policy
rules are invented and marked as such. They contain no personal information or
real customer feedback. Their reference facts describe the frozen hypothetical
world; they are not representations about a real business or legal advice.

Assistance fact paths are checked directly against the KB. Inventory keys are
`size`, `colour`, `quantity` in the per-product inventory array. Every scenario
records the permitted read-only tools, visible business record IDs, required facts,
expected answer/refusal/escalation type and category. Negative/injection scenarios
must not execute purchases, payments, account actions or network access.
