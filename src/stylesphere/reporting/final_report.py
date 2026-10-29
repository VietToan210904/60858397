"""Rebuild the requested four-pair synthesis from retained empirical evidence."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

from stylesphere.common.io import read_jsonl, sha256_file, utc_now, write_json
from stylesphere.common.statistics import paired_bootstrap

RUNS = {
    'fashion_siglip':'24512250744c475fb358342058a8a043',
    'siglip2':'24512250744c475fb358342058a8a043',
    'sdxl_lightning':'c72db9e579f046999779f6fba62b7849',
    'flux2_klein':'9a379c1a438b447a871f4c16f5e78417',
    'phi4_mini':'d9923b2e32d54eb0955cbc11cc801808',
    'qwen35_4b':'ae13ded36915404e929fa8428f956235',
    'sf3d':'7685fb6f464d4360a1ebadb8b1994d49',
    'spar3d':'d088458f86e447f58e6c81f257cbb027',
}
NAMES={'fashion_siglip':'FashionSigLIP','siglip2':'SigLIP 2','sdxl_lightning':'SDXL-Lightning',
       'flux2_klein':'FLUX.2 Klein FP8','phi4_mini':'Phi-4-mini','qwen35_4b':'Qwen3.5-4B',
       'sf3d':'SF3D','spar3d':'SPAR3D'}


def build(root: Path) -> dict:
    root=Path(root).resolve()
    folder=root/'results/final_evaluation'
    folder.mkdir(exist_ok=True)
    sources={'src/stylesphere/reporting/final_report.py':sha256_file(Path(__file__))}
    def read(name):
        sources[name]=sha256_file(root/name)
        return json.loads((root/name).read_text(encoding='utf-8'))
    retrieval=read('results/retrieval/ai_assessment/v1/summary.json')
    visual=read('results/ai_visual_assessment/summary.json')
    decisions=read('results/ai_visual_assessment/decision_coverage.json')
    assistance=read('results/assistance/ai_assessment/v1/summary.json')
    reference=read('results/reference_test/reference_summary.json')
    original=[]
    for pair in ('retrieval','generation','assistance','reconstruction'):
        path=f'results/{pair}/records.jsonl'
        sources[path]=sha256_file(root/path)
        original.extend(r for r in read_jsonl(root/path) if r.get('run_id')==RUNS.get(r.get('model')))
    if len(original)!=584:
        raise ValueError('Expected exactly 584 original final attempts, excluding smokes/retries.')
    performance=[]
    for model in RUNS:
        rows=[r for r in original if r['model']==model]
        pair=rows[0]['pair']
        key='query_inference_s' if pair=='retrieval' else 'end_to_end_s' if pair=='assistance' else 'inference_s'
        def numeric(name):
            return [r.get('timings',{}).get(name,r.get('metrics',{}).get(name)) for r in rows
                    if isinstance(r.get('timings',{}).get(name,r.get('metrics',{}).get(name)),(int,float))]
        performance.append({'pair':pair,'model':model,'run_id':RUNS[model], 'attempts':len(rows),
            'completed':sum(r['status']=='success' for r in rows),'failed':sum(r['status']=='failed' for r in rows),
            'timing_window':key,'mean_seconds':statistics.mean(numeric(key)),
            'max_allocated_mib':max(numeric('peak_allocated_mb')),
            'max_reserved_mib':max(numeric('peak_reserved_mb'))})
    ret={r['model']:r for r in retrieval['rows']}
    vis={(r['pair'],r['model']):r for r in visual['models']}
    ass={r['model']:r for r in assistance['rows'] if r['category']=='all'}
    packets=read_jsonl(root/'results/assistance/ai_assessment/v1/evidence_packets.jsonl')
    assessments=read_jsonl(root/'results/assistance/ai_assessment/v1/assessments.jsonl')
    evidence={r['assessment_id']:r for r in packets}
    by_model={m:{r['case_id']:r for r in assessments if r['model']==m} for m in ass}
    paired=[]
    cases=sorted(by_model['phi4_mini'])
    for metric in ('semantic_task_success','operational_completion','operational_grounded_semantic_success'):
        def value(row):
            p=evidence[row['assessment_id']]
            semantic=row['ai_semantic_task_verdict']=='satisfied'
            operational=p['deterministic']['protocol_completed']
            return int(semantic if metric=='semantic_task_success' else operational if metric=='operational_completion'
                       else semantic and operational and row['ai_ungrounded_claim_present'] is False)
        clusters=[evidence[by_model['phi4_mini'][c]['assessment_id']]['scenario']['evaluation_category'] for c in cases]
        paired.append({'metric':metric,'difference':'qwen35_4b_minus_phi4_mini','n_matched_attempts':80,
                       **paired_bootstrap([value(by_model['qwen35_4b'][c]) for c in cases],
                                          [value(by_model['phi4_mini'][c]) for c in cases],clusters=clusters)})
    data={'generated_at_utc':utc_now(),'requested_scope':'four pairs / eight models, Evaluation Codes and Data',
          'original_attempts':584,'original_completed':511,'original_protocol_failures':73,
          'performance':performance,'retrieval':retrieval,'visual':visual,'shortlist_decisions':decisions,
          'assistance_all_attempts':list(ass.values()),'assistance_paired_category_bootstrap':paired,
          'reference_test':reference,'source_sha256':sources}
    write_json(folder/'summary.json',data)
    with (folder/'deployment_results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(performance[0]))
        writer.writeheader();writer.writerows(performance)
    with (folder/'assistance_paired_comparisons.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]))
        writer.writeheader();writer.writerows(paired)
    def rm(model,metric,scenario='known_labels_unknown_zero_gain'):
        return ret[model]['metrics'][scenario][metric]['mean']
    def vm(pair,model,metric):
        return vis[(pair,model)]['metrics'][metric]['mean']
    def fm(model,metric):
        return reference['models'][model]['mean_metrics'][metric]
    text='# StyleSphere: final four-pair evaluation\n\n'
    text+='The confirmed scope is four pairs/eight exact models and the Evaluation Codes and Data component. AI use was expressly permitted by the user. This package contains actual local CUDA runs, preserved failures, quantitative analysis, qualitative evidence and reproduction instructions. Completion of that component does not imply independent human validation or unrestricted commercial readiness.\n\n'
    text+='The original 584 planned attempts produced 511 technical completions and 73 shopping-protocol failures. A separate reference extension adds 40 test and eight development reconstructions. Generation, retrieval and shopping quality supplements are **AI assessments**, labelled separately from unchanged human worksheets and original raw metrics. The reference meshes are licensed real scans; their conditioning images are controlled renders.\n\n'
    text+='![Observed quality under the declared protocols; different evidence types and units by panel.](quality_overview.png)\n\n'
    text+='## Local deployment evidence\n\n| Task | Model | Completed / attempts | Mean measured window | Max allocated / reserved MiB |\n|---|---|---:|---:|---:|\n'
    for row in performance:
        text+=f"|{row['pair']}|{NAMES[row['model']]}|{row['completed']}/{row['attempts']}|{row['mean_seconds']:.6f}s ({row['timing_window']})|{row['max_allocated_mib']:.2f} / {row['max_reserved_mib']:.2f}|\n"
    text+='\nThe measured workstation used Windows and an RTX 4070 SUPER with 12 GB VRAM, official pinned checkpoints and model-native preprocessing. Timings compare the stated deployment windows, not architecture alone. The 3D worker model-load timer is not full process startup; SPAR3D low-VRAM mode includes offloading/module construction within case inference. Other load, export and indexing windows are retained separately. Qwen\'s peak reserved allocator counter exceeds physical VRAM; it is an unmodified allocator observation, not proof of physical GPU residency. The selected original final runs recorded no OOM. Source/runtime/checkpoint manifests identify exact versions; system drivers/toolkits were left unchanged.\n\n'
    text+='## Product retrieval\n\n600 real ABO photos and 120 controlled queries were shared by both encoders. AI reviewers inspected all 600 photos plus exact multilingual source facts and froze product profiles before opening saved model rankings. The frozen rules then derived 72,000 proxy grades without using predictions to assign labels: 71,780 grades are known and 220 are uncertain. Useful relevance means grade >= 2; graded gain is `2**grade - 1`.\n\n| AI proxy metric | FashionSigLIP | SigLIP 2 | Eligible queries |\n|---|---:|---:|---:|\n'
    for key,label in [('ndcg_at_10','nDCG@10'),('mrr_at_10','MRR@10'),('hit_at_10','Hit@10'),('precision_at_10','Precision@10'),('recall_at_10','Known-proxy-relevant Recall@10')]:
        text+=f"|{label}|{rm('fashion_siglip',key):.6f}|{rm('siglip2',key):.6f}|{ret['fashion_siglip']['metrics']['known_labels_unknown_zero_gain'][key]['n_assessable']}|\n"
    comparison=next(c for c in retrieval['paired_comparisons'] if c['scenario']=='known_labels_unknown_zero_gain' and c['metric']=='ndcg_at_10')
    ci=comparison['cluster_bootstrap_95_percentile_interval']
    text+=f"\nSigLIP 2 minus FashionSigLIP nDCG difference is {comparison['paired_mean_difference']:+.6f}, with a 95% interval [{ci[0]:.6f}, {ci[1]:.6f}], resampling 12 query families. This does not establish a general quality winner. Mean top-10 judged coverage is {ret['fashion_siglip']['mean_judged_coverage_at_10']:.2%}/{ret['siglip2']['mean_judged_coverage_at_10']:.2%}; only 50/120 queries have complete catalogue proxy judgements. Main scores assign unknowns zero gain; all-unknown-grade-3 sensitivity nDCGs are {rm('fashion_siglip','ndcg_at_10','all_unknown_grade3_sensitivity'):.6f}/{rm('siglip2','ndcg_at_10','all_unknown_grade3_sensitivity'):.6f} over 119 queries. Conservative bounds and every query are retained. Five main nDCGs have no known graded-positive item; 16 recalls have no known useful-positive item and remain null. These are AI/source proxies, not human qrels. Running suitability and lightweight/travel are appearance proxies, with explicit source exceptions, not measured product performance.\n\n"
    text+='Recommendation: FashionSigLIP is a resource-efficient pilot candidate on this catalogue; its main nDCG/MRR point estimates are slightly higher with lower observed query latency and memory. Keep SigLIP 2 as a credible alternative pending representative customer queries and independent relevance validation.\n\n'
    text+='## Campaign generation\n\nBoth models generated the same 24 briefs with three seeds each, at 1024 × 1024 resolution and four native steps. All 144 images were assessed in shuffled model-masked packets after a separate AI rubric freeze. Some earlier QA had been seen, so this is post-hoc supplementary review, not an independent blinded human study.\n\n| AI ordinal score (1–5) | SDXL-Lightning | FLUX.2 Klein FP8 |\n|---|---:|---:|\n'
    for key,label in [('ai_prompt_adherence','Prompt adherence'),('ai_visual_coherence','Visual coherence'),('ai_composition','Composition'),('ai_usability_rating','Concept usability')]:
        text+=f"|{label}|{vm('generation','sdxl_lightning',key):.4f}|{vm('generation','flux2_klein',key):.4f}|\n"
    text+='\nOrdinal means weight 24 prompts equally; three seeds are nested within each prompt. FLUX minus SDXL adherence is +1.3889 points, with a 95% prompt-bootstrap interval [1.0000, 1.7917]. AI shortlist decisions are 40 yes/19 no/13 uncertain for FLUX and 19 yes/52 no/one uncertain for SDXL. Possible positive-rate bounds across all 72 planned images are 55.56–73.61% and 26.39–27.78%; these are unresolved-decision scenarios, not imputed outcomes. Conditional shortlist prompt means use 23 versus 24 prompts and must not be described as equal-denominator approval rates. The 58 complete binary pairs cover 23 prompts; only 16 prompts have all six decisions. See denominator diagnostics for each analysis.\n\n'
    text+='Recommendation: FLUX is the stronger candidate for further design review on these briefs under the AI adherence/usability rubric, at higher measured latency and memory cost. SDXL is a faster draft candidate when correction is acceptable. Neither result authorises publishing an image or establishes consumer preference. Concrete successes, missed attributes and the preserved assessment caveat are in [qualitative findings](../../docs/QUALITATIVE_FINDINGS.md).\n\n'
    text+='## Grounded shopping assistance\n\nThe same 80 hypothetical scenarios, frozen knowledge base, read-only tool allowlist and strict six-turn controller were used. All 160 retained outputs were read against requests, policy facts and tool traces. An understandable answer in invalid JSON can receive semantic credit while remaining an operational failure.\n\n| Outcome | Phi-4-mini | Qwen3.5-4B |\n|---|---:|---:|\n'
    for label,key in [('Original protocol completions','original_protocol_completions'),('AI semantic task successes','ai_semantic_task_success'),('Operational + grounded + semantic successes','ai_operational_and_grounded_semantic_success')]:
        values=[ass[m][key]['positive'] if isinstance(ass[m][key],dict) else ass[m][key] for m in ('phi4_mini','qwen35_4b')]
        text+=f'|{label}|{values[0]}/80|{values[1]}/80|\n'
    text+='|AI factual prose supported|51/71 assessable|52/68 assessable|\n|AI unsupported claims present|20/78 prose outputs|16/72 prose outputs|\n'
    # Use actual denominator-bearing values rather than fixed display assumptions.
    old='|AI unsupported claims present|20/78 prose outputs|16/72 prose outputs|'
    v=[ass[m]['ai_unsupported_claim_present'] for m in ('phi4_mini','qwen35_4b')]
    text=text.replace(old,f"|AI unsupported claims present|{v[0]['positive']}/{v[0]['n_assessable']} assessable|{v[1]['positive']}/{v[1]['n_assessable']} assessable|")
    text+='\nQwen delivered more AI-assessed satisfactory answers, but both achieved only 25/80 when operational validity, semantic delivery and retrieved grounding were required together. Eight category clusters support the additional paired sensitivity intervals; these responses are not independent human participants. Policy contradictions include Qwen describing AUD 120 as below an AUD 100 free-shipping threshold and rejecting a 20-day return while stating 30 days; Phi sometimes assigns faulty-return postage to the customer and repeats injected false price/policy claims. Full excerpts and exact KB paths are retained.\n\n'
    text+='Recommendation: neither deployment is ready for unattended customer support under this controller. Phi has lower measured latency and allocator peaks in this run; Qwen offers more AI-assessed semantic answers without a higher grounded-operational success count. A next controlled experiment should harden structured output and grounding, then rerun a new frozen protocol; it must not erase these failures. No monetary operating-cost measurement was made.\n\n'
    text+='## Single-image reconstruction\n\nThe original 20 real ABO photo inputs produced 40 GLBs: SF3D strict-valid 19/20 and SPAR3D 20/20, with one degenerate SF3D face retained. AI preview means are higher for SF3D than SPAR3D for visible fidelity (3.15/2.70), shape (3.10/2.40), texture (2.45/2.15) and multiview coherence (3.25/2.55). These are AI opinions about ABO previews, separate from GSO reference scores. Hidden-surface correctness remains uncertain for all 40 because those original photos lack aligned references.\n\n'
    text+='The separate controlled-reference extension uses 20 licensed real Google Scanned Objects meshes: 17 shoes and three backpacks, with four disjoint development objects excluded from test results. Both models completed all 20 test inputs. Common RGBA inputs and four held-out reference views were rasterized from scans. Source selection, cameras, normalization, sampling, 24 proper-rotation ICP initializations and the 50-iteration limit were fixed before development/test inference. Measurement uses 10,000 surface points independent of 2,048 registration points.\n\n| Dimensionless reference metric | SF3D | SPAR3D |\n|---|---:|---:|\n'
    for key,label in [('unaligned_chamfer_squared','Normalized, no fitted pose: squared Chamfer↓'),('rigid_aligned_chamfer_squared','Reference-assisted rigid alignment: squared Chamfer↓'),('unaligned_fscore_at_0.02','Normalized, no fitted pose: F-score@0.02↑'),('rigid_aligned_fscore_at_0.01','Reference-assisted rigid alignment: F-score@0.01↑'),('rigid_aligned_fscore_at_0.02','Reference-assisted rigid alignment: F-score@0.02↑'),('rigid_aligned_supplementary_silhouette_iou','Aligned supplementary silhouette IoU↑')]:
        text+=f"|{label}|{fm('sf3d',key):.6f}|{fm('spar3d',key):.6f}|\n"
    c=reference['paired_differences_spar3d_minus_sf3d']['rigid_aligned_fscore_at_0.02']
    text+=f"\nSPAR3D minus SF3D aligned F-score@0.02 difference is {c['mean']:+.6f}, with a category-stratified object-bootstrap 95% interval [{c['ci95_low']:.6f}, {c['ci95_high']:.6f}], over 20 matched objects. Full unaligned/aligned values, per-object transformations, sampled surfaces, masks and intervals are retained. Independent bounding-box-diagonal normalization removes physical size/translation and is orientation-dependent. ICP uses reference geometry to remove nuisance pose, with no reflection or further fitted scale. These are shape scores under that procedure, not absolute metric reconstruction accuracy.\n\n"
    text+='Recommendation: SF3D is a candidate for a resource-conscious preview pilot, with lower mean Chamfer, higher supplementary silhouette IoU and approximately 19-fold shorter original ABO inference windows. The quality evidence is mixed: SPAR3D has the higher aligned F-score at the fine 0.01 threshold and a slightly higher F-score at 0.02 whose paired interval crosses zero. No model dominates every shape measure. Treat both as approximate assets needing inspection. This controlled set is ASICS-heavy, often end-on, uses rendered rather than natural conditioning images and has unknown overlap with model training data. Native cameras/crops differ across models; original photo preview centering is not camera calibration. No physical dimensions, hidden-view accuracy on ABO, watertightness repair or manufacturability is established.\n\n'
    text+='## Reproduction and evidence limits\n\nAI reviewers were implementation contributors, not independent human participants; the serving judge version was not exposed. Review uncertainty, source conflicts and missing decisions are preserved. Bootstrap intervals describe variation among these designed prompts/query families/scenarios/objects, conditional on fixed AI labels; they do not estimate judge bias, real-customer preferences or general deployment risk. No inter-rater agreement or customer feedback is invented.\n\n'
    text+='Start with [the README](../../README.md) and [reproduction commands](../../docs/REPRODUCIBILITY.md). The submission includes data licences/attribution, source assets permitted for redistribution, frozen inputs, raw outputs/failures, hashes, dependency/build receipts, tests, quality ledgers and paired results. Weights, virtual environments, caches and credentials are excluded. Recompute summaries without model inference; rerunning the eight models requires the pinned official downloads and local environments. Observed native execution and same-workstation resume were verified; a cold rebuild on another workstation has not been verified and requires a new deployment freeze. The legacy verifier keeps independent-human/absolute-reference requirements separate from completion of the confirmed component.\n'
    (folder/'REPORT.md').write_text(text,encoding='utf-8')
    plot(folder,rm,vm,ass,fm)
    manifest={'generated_at_utc':utc_now(),'source_sha256':sources,
              'generated_sha256':{p.name:sha256_file(p) for p in folder.iterdir() if p.is_file() and p.name not in {'manifest.json','completion.json'}}}
    write_json(folder/'manifest.json',manifest)
    return {'report':str(folder/'REPORT.md'),'original_attempts':len(original),'reference_test_attempts':reference['raw_attempts']}


def plot(folder,rm,vm,ass,fm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    panels=[('AI proxy retrieval nDCG@10 (115 queries)', ['FashionSigLIP','SigLIP2'],[rm('fashion_siglip','ndcg_at_10'),rm('siglip2','ndcg_at_10')],1),
            ('AI generation adherence (24 prompts)', ['SDXL','FLUX FP8'],[vm('generation','sdxl_lightning','ai_prompt_adherence'),vm('generation','flux2_klein','ai_prompt_adherence')],5),
            ('Grounded + operational + AI semantic success', ['Phi-4-mini','Qwen3.5'],[ass[m]['ai_operational_and_grounded_semantic_success']/80 for m in ('phi4_mini','qwen35_4b')],1),
            ('Reference-assisted aligned F-score@0.02 (20 objects)', ['SF3D','SPAR3D'],[fm('sf3d','rigid_aligned_fscore_at_0.02'),fm('spar3d','rigid_aligned_fscore_at_0.02')],1)]
    for ax,(title,labels,values,top) in zip(axes.flat,panels):
        bars=ax.bar(labels,values,color=['#0072B2','#D55E00'],width=.55)
        ax.set(title=title,ylim=(0,top));ax.spines[['right','top']].set_visible(False)
        ax.bar_label(bars,fmt='%.3f',padding=3)
    fig.suptitle('StyleSphere: observed quality under the declared assessment protocols',fontsize=13)
    fig.supxlabel('Different units and evidence types per panel. AI opinions/proxies are not human validation; uncertainty is in the paired tables.',fontsize=9)
    for extension in ('png','svg'):
        fig.savefig(folder/f'quality_overview.{extension}',dpi=180)
    plt.close(fig)


def validate_report_snapshot(root: Path) -> list[str]:
    """Reject a final synthesis whose source data or displayed artifacts changed."""
    root=Path(root).resolve()
    folder=root/'results/final_evaluation'
    errors=[]
    try:
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if not {'REPORT.md','summary.json','deployment_results.csv','quality_overview.png'} <= set(manifest['generated_sha256']):
            raise ValueError('Final synthesis manifest lacks required artifacts')
        for base,bindings in ((root,manifest['source_sha256']),(folder,manifest['generated_sha256'])):
            for name,digest in bindings.items():
                path=(base/name).resolve()
                if not path.is_relative_to(base) or not path.is_file() or sha256_file(path)!=digest:
                    errors.append(f'Final report snapshot differs: {name}')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        errors.append(f'Final report snapshot: {exc}')
    return errors


if __name__=='__main__':
    print(json.dumps(build(Path.cwd())))
