# Implementation and execution status

Evidence date: 29 October 2026, Australia/Sydney. The user confirmed the original
four pairs/eight exact models and expressly permitted AI assistance. The requested
Evaluation Codes and Data component is complete, subject to the validated receipt
in `results/final_evaluation/completion.json`. Completion means the stated local
experiments, separate AI analyses, reference experiment, tests, documentation and
package were executed; it does not invent independent human feedback.

| Component | Status | Actual evidence |
|---|---|---|
| Core CLI, schemas, metrics, recording, provenance, resumability | IMPLEMENTED / LOCALLY TESTED | Source, tests, environment receipts and raw artifacts retained |
| FashionSigLIP vs SigLIP2 | EVALUATED under declared AI proxy protocol | Same 600 ABO photos and 120 queries; 240 local CUDA successes; all 600 photos reviewed before rankings; 72,000 proxy grades including 220 unknowns; all 240 rankings scored |
| SDXL-Lightning vs FLUX.2 Klein FP8 | EVALUATED under declared AI visual protocol | 72 real 1024-square images each; all 144 reviewed; ordinal means, prompt-cluster comparisons, explicit binary decision uncertainty |
| Phi-4-mini vs Qwen3.5-4B | EVALUATED under frozen controller and separate AI semantic review | Same 80 scenarios each; 40/47 protocol completions and 40/33 retained failures; all 160 outputs reviewed, including failures |
| SPAR3D vs SF3D, original ABO photos | EVALUATED for execution, topology and AI visible quality | 20 exports each, no execution failures; strict mesh validity SF 19/20, SP 20/20; 40 outputs with four previews each and AI review |
| SPAR3D vs SF3D, real scanned references | EVALUATED under controlled normalized-shape protocol | 20 disjoint test references and 4 development objects; 48 local CUDA exports; all geometry/mask scores verified; test only 20 paired objects; resume proof |
| Final synthesis, recommendation tables, figures and reproducible package | IMPLEMENTED / VERIFIED | Final report, deployment CSV, per-case metrics, hashes and extracted-package verification |
| Independent human/customer feedback | NOT COLLECTED / NOT CLAIMED | Human worksheets remain blank; AI opinions are explicitly separate |
| Absolute physical 3D accuracy | NOT CLAIMED | Original ABO lacks aligned references; new shape analysis removes size/translation and labels reference-assisted rigid alignment |

No selected model remains blocked. The original 584 planned attempts yielded
511 technical completions and 73 strict shopping-protocol failures; those failures
are empirical findings, not unfinished executions. The reference extension adds
40 test and 8 development executions, all successful. Original raw records and
frozen inference workers remain unchanged.

Read `results/final_evaluation/REPORT.md` for conclusions and denominators,
`docs/QUALITATIVE_FINDINGS.md` for inspected examples, and
`docs/REPRODUCIBILITY.md` for commands. Key separate evidence is in:

- `results/retrieval/ai_assessment/v1/`: pre-ranking profiles, proxy qrels, query scores and paired uncertainty.
- `results/ai_visual_assessment/`: all 184 assessments, image hashes and shortlist denominator diagnostics.
- `results/assistance/ai_assessment/v1/`: all 160 trace-bound decisions and policy evidence.
- `results/reference_test/`: 40 new raw outputs, normalized and aligned geometry metrics, sample clouds and calibrated reference masks.
- `environment/reference_geometry_audit.json` and `reference_resume_audit.json`: independent recomputation and unchanged-record resumes.
- `submission/package-result.json` and `extraction-validation.json`: final archive identity and fresh-extraction checks.

The exact official model revisions are in `config/models.yaml`. Actual precision,
CUDA allocator counters, dependency/native-build identities and timings are in
raw records and `environment/`. SPAR3D uses this deployment's low-VRAM mode;
cross-model timings therefore compare tested deployments, not architecture alone.
Qwen's reserved allocator counter exceeds physical VRAM; no physical-residency or
paging-cause claim is made. No cloud model inference, substituted checkpoints,
driver/toolkit changes or fabricated human ratings were used.

The observed native Windows runtimes and same-workstation resumes were verified.
A cold rebuild on another workstation has not been verified. Rebuilding with new
native binaries or dependencies requires a separate new deployment freeze rather
than changing the included receipts.
