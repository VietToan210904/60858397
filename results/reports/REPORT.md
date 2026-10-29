# Observed execution report

Raw records: 928. Distinct case/configuration records: 928.

For the confirmed four-pair component's completed quality analyses, real-reference extension and recommendations, read [the final evaluation](../final_evaluation/REPORT.md). This file preserves the original execution and human-review reporting scope.

Blocked counts include preflight sentinel records when no dataset could be loaded; these are not executed model cases or model-failure observations. Model-load and catalogue-indexing windows are counted once per actual load/build.

Execution rates retain all logged attempts, including failed retries. Throughput is available only when every executed attempt has a measured inference window; timed/untimed coverage is explicit in model_summary.csv. failures.csv retains every raw failure/blocker. The case table shows retained resumable case outcomes; paired reliability uses the first executed attempt per matched case so a later retry cannot erase the original failure. Quality/resource comparisons use matched retained successful outputs.

Execution success and resource measurements do not establish task quality. Quality outcomes with missing labels or human assessments remain unavailable. No model winner or business recommendation is automatically produced.

Execution exception counts are in execution_error_summary.csv, separated by mode, protocol and blocked/failed status. These are observed execution diagnostics; semantic error categories still require independent review.

| Pair | Model | Mode | Protocol | Successful | Attempted | Blocked |
|---|---|---|---|---:|---:|---:|
| assistance | phi4_mini | smoke | e50d9c7557e9 | 0 | 0 | 2 |
| assistance | qwen35_4b | smoke | e50d9c7557e9 | 0 | 0 | 2 |
| assistance | phi4_mini | evaluation | e50d9c7557e9 | 0 | 0 | 80 |
| assistance | qwen35_4b | evaluation | e50d9c7557e9 | 0 | 0 | 80 |
| assistance | phi4_mini | smoke | e4981f2011b7 | 0 | 0 | 2 |
| assistance | qwen35_4b | smoke | e4981f2011b7 | 0 | 0 | 2 |
| assistance | phi4_mini | smoke | 1920e64faad5 | 1 | 2 | 0 |
| assistance | phi4_mini | smoke | 5947cc4bbeae | 2 | 2 | 0 |
| assistance | qwen35_4b | smoke | 5947cc4bbeae | 2 | 2 | 0 |
| assistance | phi4_mini | evaluation | 405c050674d1 | 40 | 80 | 0 |
| assistance | qwen35_4b | evaluation | 405c050674d1 | 47 | 80 | 0 |
| generation | sdxl_lightning | smoke | e50d9c7557e9 | 0 | 0 | 1 |
| generation | flux2_klein | smoke | e50d9c7557e9 | 0 | 0 | 1 |
| generation | sdxl_lightning | evaluation | e50d9c7557e9 | 0 | 0 | 72 |
| generation | flux2_klein | evaluation | e50d9c7557e9 | 0 | 0 | 72 |
| generation | sdxl_lightning | smoke | e4981f2011b7 | 0 | 0 | 1 |
| generation | flux2_klein | smoke | e4981f2011b7 | 0 | 0 | 1 |
| generation | sdxl_lightning | smoke | 7db8793058b6 | 1 | 1 | 0 |
| generation | sdxl_lightning | evaluation | 36989e9326c8 | 72 | 72 | 0 |
| generation | flux2_klein | smoke | 7db8793058b6 | 1 | 1 | 0 |
| generation | flux2_klein | evaluation | 36989e9326c8 | 72 | 72 | 0 |
| reconstruction | spar3d | smoke | e50d9c7557e9 | 0 | 0 | 1 |
| reconstruction | sf3d | smoke | e50d9c7557e9 | 0 | 0 | 1 |
| reconstruction | spar3d | evaluation | e50d9c7557e9 | 0 | 0 | 1 |
| reconstruction | sf3d | evaluation | e50d9c7557e9 | 0 | 0 | 1 |
| reconstruction | spar3d | smoke | e4981f2011b7 | 0 | 0 | 1 |
| reconstruction | sf3d | smoke | e4981f2011b7 | 0 | 0 | 1 |
| reconstruction | sf3d | smoke | 19fd5179afc3 | 1 | 1 | 0 |
| reconstruction | spar3d | smoke | 19fd5179afc3 | 1 | 1 | 0 |
| reconstruction | sf3d | evaluation | 59d2dda9fd31 | 20 | 20 | 0 |
| reconstruction | spar3d | evaluation | 59d2dda9fd31 | 20 | 20 | 0 |
| retrieval | fashion_siglip | smoke | e50d9c7557e9 | 0 | 0 | 1 |
| retrieval | siglip2 | smoke | e50d9c7557e9 | 0 | 0 | 1 |
| retrieval | fashion_siglip | evaluation | e50d9c7557e9 | 0 | 0 | 1 |
| retrieval | siglip2 | evaluation | e50d9c7557e9 | 0 | 0 | 1 |
| retrieval | fashion_siglip | smoke | 954d8b90a9a4 | 4 | 4 | 0 |
| retrieval | siglip2 | smoke | 954d8b90a9a4 | 4 | 4 | 0 |
| retrieval | fashion_siglip | evaluation | 19b5249ee23a | 120 | 120 | 0 |
| retrieval | siglip2 | evaluation | 19b5249ee23a | 120 | 120 | 0 |

## Matched comparisons

Values are A minus B; sign alone is not a quality recommendation. CI resamples the stated independent case/prompt/object clusters. Reliability includes executed failures; other comparisons are conditional on matched successful cases with that metric. See paired_comparisons.csv for eligible counts and confidence levels.

| Pair | A / B | Metric | Matched cases | Clusters | Mean difference | CI |
|---|---|---|---:|---:|---:|---|
| assistance | phi4_mini / qwen35_4b | execution_success_rate | 80 | 80 | -0.0875 | [-0.25, 0.075] |
| generation | flux2_klein / sdxl_lightning | execution_success_rate | 72 | 24 | 0 | [0, 0] |
| reconstruction | sf3d / spar3d | execution_success_rate | 20 | 20 | 0 | [0, 0] |
| retrieval | fashion_siglip / siglip2 | execution_success_rate | 120 | 120 | 0 | [0, 0] |
| assistance | phi4_mini / qwen35_4b | structured_task_success | 21 | 21 | -0.0952381 | [-0.238095, 0] |
| assistance | phi4_mini / qwen35_4b | factual_correctness | 18 | 18 | -0.0833333 | [-0.25, 0.0555556] |
| assistance | phi4_mini / qwen35_4b | unsupported_claim_rate | 18 | 18 | 0.0833333 | [-0.0555556, 0.25] |
| assistance | phi4_mini / qwen35_4b | required_fact_coverage | 19 | 19 | -0.0789474 | [-0.236842, 0.0526316] |
| assistance | phi4_mini / qwen35_4b | tool_call_schema_validity | 15 | 15 | 0 | [0, 0] |
| assistance | phi4_mini / qwen35_4b | tool_selection_appropriateness | 21 | 21 | -0.142857 | [-0.285714, 0] |
| assistance | phi4_mini / qwen35_4b | tool_call_count | 21 | 21 | -0.142857 | [-0.382143, 0.0952381] |
| assistance | phi4_mini / qwen35_4b | inference_s | 21 | 21 | -4.05757 | [-7.28302, -2.0703] |
| assistance | phi4_mini / qwen35_4b | peak_allocated_mb | 21 | 21 | -1437.96 | [-1732.8, -1247.23] |
| assistance | phi4_mini / qwen35_4b | peak_reserved_mb | 21 | 21 | -2219.52 | [-2821.81, -1747.71] |
| assistance | phi4_mini / qwen35_4b | end_to_end_s | 21 | 21 | -4.05875 | [-7.28535, -2.07077] |
| generation | flux2_klein / sdxl_lightning | peak_allocated_mb | 72 | 24 | 3959.52 | [3955.52, 3962.86] |
| generation | flux2_klein / sdxl_lightning | peak_reserved_mb | 72 | 24 | 2739.06 | [2732, 2747.06] |
| generation | flux2_klein / sdxl_lightning | valid_output | 72 | 24 | 0 | [0, 0] |
| generation | flux2_klein / sdxl_lightning | inference_s | 72 | 24 | 1.85607 | [1.79264, 1.90855] |
| reconstruction | sf3d / spar3d | peak_allocated_mb | 20 | 20 | -1888.96 | [-1890, -1887.76] |
| reconstruction | sf3d / spar3d | peak_reserved_mb | 20 | 20 | -1137.8 | [-1140.6, -1134.2] |
| reconstruction | sf3d / spar3d | glb_export_success | 20 | 20 | 0 | [0, 0] |
| reconstruction | sf3d / spar3d | mesh_valid | 20 | 20 | -0.05 | [-0.15, 0] |
| reconstruction | sf3d / spar3d | mesh_watertight | 20 | 20 | 0 | [0, 0] |
| reconstruction | sf3d / spar3d | inference_s | 20 | 20 | -12.0314 | [-12.3742, -11.7884] |
| reconstruction | sf3d / spar3d | end_to_end_s | 20 | 20 | -12.0345 | [-12.3795, -11.7897] |
| retrieval | fashion_siglip / siglip2 | query_inference_s | 120 | 120 | -0.000524346 | [-0.000683436, -0.000356006] |
| retrieval | fashion_siglip / siglip2 | end_to_end_s | 120 | 120 | -0.00213253 | [-0.00230388, -0.00194176] |
| retrieval | fashion_siglip / siglip2 | peak_allocated_mb | 120 | 120 | -656.627 | [-656.627, -656.627] |
| retrieval | fashion_siglip / siglip2 | peak_reserved_mb | 120 | 120 | -678 | [-678, -678] |

## Figure captions

- assistance_end_to_end_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_execution_success_rate: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_factual_correctness: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_inference_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_model_load_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_peak_allocated_mb: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_peak_reserved_mb: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_required_fact_coverage: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_structured_task_success: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_tool_call_count: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_tool_call_schema_validity: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_tool_selection_appropriateness: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- assistance_unsupported_claim_rate: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- generation_execution_success_rate: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- generation_inference_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- generation_model_load_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- generation_peak_allocated_mb: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- generation_peak_reserved_mb: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- generation_valid_output: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- generation_valid_outputs_per_inference_second: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- reconstruction_end_to_end_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- reconstruction_execution_success_rate: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- reconstruction_glb_export_success: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- reconstruction_inference_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- reconstruction_mesh_valid: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- reconstruction_mesh_watertight: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- reconstruction_model_load_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- reconstruction_peak_allocated_mb: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- reconstruction_peak_reserved_mb: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- retrieval_catalogue_indexing_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- retrieval_end_to_end_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- retrieval_execution_success_rate: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- retrieval_model_load_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- retrieval_peak_allocated_mb: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- retrieval_peak_reserved_mb: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
- retrieval_query_inference_s: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.
