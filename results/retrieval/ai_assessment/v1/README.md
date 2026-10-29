# Supplementary AI visual/source retrieval assessment

This supplement contains **AI proxy relevance labels, not human qrels or independently validated ground truth**. Three Codex coding assistants reviewed all 600 catalogue photos and exact cached ABO source facts. They contributed to the implementation and could see model identities; predicted rankings were withheld until all product annotations were frozen. No independent human validation or inter-rater reliability estimate is claimed. Product batches were disjoint, so rater effects were not measured.

The rubric was accepted and frozen before rating. The 600 profiles were frozen at `2026-10-28T23:50:51Z`, SHA256 `fca0655f560531c8bbba093fe159e3ceebb1eb00baf97a76ea1f196100ba4d6c`. Only then were the 240 saved rankings from original run `24512250744c475fb358342058a8a043` opened for scoring. No original model inference was rerun. Catalogue, queries, human qrels, model outputs and frozen runtime remain unchanged.

Each product has authored type labels, ten explicit visual-trait ratings, confidence and an actual image note. Type labels follow the photograph and exact source facts rather than blindly copying broad source categories: knee-high boots are not forced into ankle boots; bracelets are not forced into necklaces; gaiters are not forced into hats. Ambiguity remains explicit. Exact source citations retain original language tags and values. The cached multilingual source supplies names for 600/600 products, compared with 402 English names; colour coverage is 460 versus 297 and style coverage 380 versus 227. Unknown properties are not invented.

The deterministic rules produce all **72,000 query/product pairs**: definite wrong type or absent/conflicting trait gives0; weak/partial visual match gives1; strong match gives3. Ordinary traits intentionally have no grade2. For “lightweight ... for travel”, a strong travel-appearance proxy gives2, with3 permitted only when an exact source citation explicitly supports lightweight. Only profiles238 and523 have that source support. Actual weight, comfort, running performance and travel suitability were not measured. “Running shoes” denotes visible athletic/running-shoe appearance or explicit source identification. Formality, minimalism and everyday suitability remain subjective style judgements.

There are 71,780 known grades and220 unknown grades, arising from22 product-type uncertainties. Fifty of120 queries have complete catalogue labels. Unknowns remain null in `proxy_qrels.jsonl`. Main metrics retain original ranking positions and give unknown entries zero gain; this is an explicitly labelled conditional scenario, not relabelling unknown products as negatives. The separate all-unknown-grade3 scenario tests sensitivity. Conservative nDCG bounds are also supplied using known DCG divided by potential ideal DCG, and potential DCG divided by known ideal DCG, capped at1. A regression enumerates possible unknown assignments to check those bounds.

|Main known-label scenario|FashionSigLIP|SigLIP2|Assessed queries|
|---|---:|---:|---:|
|AI proxy nDCG@10|0.7823|0.7649|115/120|
|AI proxy MRR@10, useful grade>=2|0.7711|0.7342|120/120|
|AI proxy Hit@10, useful grade>=2|0.8667|0.8667|120/120|
|AI proxy precision@10, useful grade>=2|0.4917|0.4808|120/120|
|Recall@10 against known AI-proxy useful items|0.5299|0.5328|104/120|
|Mean judged coverage of top10|97.83%|96.92%|120/120|

Five queries have no known positive graded gain, so their main nDCG is undefined. Sixteen have no known grade>=2 item, so known-relevant recall is undefined. Their Hit/MRR values remain0 and are included; catalogue suitability affects these results as well as ranking quality. Denominators and per-query coverage are retained rather than filling missing quality values.

Under the all-unknown-grade3 sensitivity scenario, mean nDCG is0.6885 versus0.6805 over119 assessable queries; Hit@10 is0.9083 versus0.9333. The two sensitivity scenarios are not themselves mathematical lower/upper nDCG bounds: changing unknown labels changes the ideal ranking too. The separate conservative bounds account for this.

Paired bootstrap resamples the **12 query-type families**, keeping their ten related variants together, with2,000 replicates and seed8410. The main nDCG difference, SigLIP2 minus FashionSigLIP, is-0.01742 with a95% percentile interval[-0.03927,0.00421]. This conditional comparison does not establish a general model winner. Neither120 variants nor600 shared product labels are independent human assessments, and the interval does not measure label accuracy or inter-rater uncertainty.

`assessment_protocol.json` and `protocol_freeze.json` preserve the rubric. `product_evidence.jsonl` retains all assessed multilingual listings, source fields and exact image hashes; `sheets/` and `source_sidecars/` contain the ranking-independent review batches. `annotations/` retains authored decisions. `frozen_product_profiles.jsonl` and `profiles_freeze.json` bind all600 profiles before score access. `proxy_qrels.jsonl`, `query_coverage.json`, `qrels_manifest.json`, `query_scores.jsonl` and `summary.json` retain every pair, query and score with source hashes. Original compressed listing cache is optional for portable recomputation because full assessed listings are retained in hashed evidence; if the cache exists, its original SHA is also checked.

Recompute the derived labels and metrics from the workspace root in PowerShell:

```powershell
& .\.venv-retrieval\Scripts\python.exe -m stylesphere.retrieval.proxy_metrics qrels
& .\.venv-retrieval\Scripts\python.exe -m stylesphere.retrieval.proxy_metrics score --run-id 24512250744c475fb358342058a8a043
& .\.venv-retrieval\Scripts\python.exe -m pytest tests/test_retrieval_ai_proxy.py -q
```

`stylesphere.retrieval.ai_assessment export` prepares review evidence without reading rankings. `proxy_metrics freeze` checks complete, unique product coverage, exact image identity, ten valid trait labels and exact multilingual citations before creating the immutable profile receipt. Re-running aggregation uses the retained AI judgements; it does not substitute for a new independent assessment. All supplementary quality fields remain separate from the original human-primary quality fields.
