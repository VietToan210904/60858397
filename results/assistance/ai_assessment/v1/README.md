# Supplementary AI review of assistance responses

These are **AI-authored judgements, not human feedback or independently established ground truth**. The Codex coding assistant read all 160 retained final-run requests, outputs, frozen business records and tool evidence, including all 73 original protocol failures. The assessor helped implement the controller, could see model identities, and has no independently validated reliability estimate. The exact backend model version is not exposed in this session. These qualifications apply to every score below.

The original benchmark records, deployment settings, traces and human review worksheet remain unchanged. No inference reruns, cloud API calls or GPU calls were made for this supplement. Its pipeline exports evidence, validates explicit authored judgements and aggregates them; it does not infer prose correctness from keywords, matching numeric tokens or structured claims.

|Measure|Phi-4-mini|Qwen3.5-4B|
|---|---:|---:|
|Original protocol completions|40/80|47/80|
|AI-assessed customer task satisfied, ignoring JSON syntax|42/80 (52.5%)|54/80 (67.5%)|
|AI-supported factual prose, among responses with factual assertions|51/71 (71.8%)|52/68 (76.5%)|
|AI-assessed unsupported assertion present, among customer responses|20/78 (25.6%)|16/72 (22.2%)|
|AI-assessed ungrounded assertion present, among customer responses|25/78 (32.1%)|21/72 (29.2%)|
|AI-assessed appropriate tool selection|38/80|59/80|
|Exact actually executed tool set matches scenario expectation|38/80|52/80|
|Original protocol failure with semantically satisfactory prose|15|24|
|Original completion AND semantic satisfaction AND no ungrounded assertion|25/80|25/80|
|No customer-facing prose|2/80|8/80|

These descriptive results support comparison under this particular deployment and rubric; they do not establish a general model winner. All 80 attempts per model remain in the semantic and operational denominators. Missing prose is a failed customer delivery, not an invented factual error or a perfect factual score. Pure promises or bare refusal labels also receive no factual-content score. Unsupported and ungrounded rates exclude only missing customer prose. Category-level counts, confidence, omissions and evidence are retained in the machine-readable files.

“Supported” compares assertions with the hypothetical frozen KB and controller. An unsupported assertion can be false, or simply unestablished by those sources; it is not automatically false in the external world. “Ungrounded” also includes a true business rule stated without retrieving the authorized record. Generic capability statements can rely on the system/controller policy. Concrete unverified business channels, policy exceptions and time-window anchors are conservatively marked unsupported, with medium confidence where wording admits alternatives. The semantic task axis judges the actual customer request separately from extra assertions. For example, a correct 30-day duration can satisfy a duration question while an added purchase-date anchor remains unsupported. A predefined checklist may contain an ancillary fact not explicitly requested; the checklist is reported separately rather than mechanically determining task success.

Readable answer strings inside malformed outer JSON are retained for review, including Qwen product-detail answers. Their original operational failures remain failures. A batch of relevant proposed tool calls may show appropriate tool choice yet produce no executed calls or customer answer. Qwen's eight tool-only terminal batches and Phi's two structured-only refusals are therefore explicit missing-prose cases. An unauthorized capability offer is a response error; no prohibited mutation was executed by the allowlisted controller.

`authored_decisions.txt` contains all 160 explicit per-case decisions and rationales. `assessments.jsonl` expands them into readable labels with exact output quotations, KB references and source hashes. `evidence_packets.jsonl` contains the actual terminal outputs, all retained turns, scenario, frozen KB, observations and original status. Every source artifact hash and original trace replay is checked before export and summary. The source manifest binds the evidence, protocol and assessment implementation. Summary also verifies that the expanded records equal the authored ledger. The full evidence duplicates business fixtures for portability; these fixtures are hypothetical.

Reproduce the validation and aggregation from the workspace root in PowerShell:

```powershell
& .\.venv-retrieval\Scripts\python.exe -m stylesphere.assistance.ai_assessment materialize
& .\.venv-retrieval\Scripts\python.exe -m stylesphere.assistance.ai_assessment summarize
& .\.venv-retrieval\Scripts\python.exe -m pytest tests/test_ai_assistance_assessment.py -q
```

The authored semantic decisions are a retained assessment artifact, not a deterministic substitute for a fresh independent assessor. Re-running these commands validates and aggregates the same decisions. `export` is the evidence preparation command; use a new version directory if inputs change. Historical source runs are explicitly selected in the implementation, so later runs cannot silently replace these 160 first final-run attempts.
