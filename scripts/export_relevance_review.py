"""Build a local human relevance worksheet from independent annotation candidates.

No model ranking is read. Grades start blank; exported CSV contains graded rows
only and requires an annotator ID. Human judgement is never inferred from text.
"""
import argparse
import csv
import json
import os
from pathlib import Path


def export(root):
    folder = Path(root).resolve() / "datasets" / "retrieval"
    with (folder / "annotation_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No annotation candidates; prepare a real catalogue and query set first")
    target = folder / "review"
    target.mkdir(exist_ok=True)
    products = {}
    with (folder / "catalogue.csv").open(encoding="utf-8-sig", newline="") as stream:
        products = {r["product_id"]: r for r in csv.DictReader(stream)}
    for row in rows:
        image = Path(root).resolve() / row["image_path"]
        row["image"] = Path(os.path.relpath(image, target)).as_posix()
        row["attributes"] = products[row["product_id"]].get("attributes", "")
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    page = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Independent relevance review</title>
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file: data:; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<style>body{font-family:system-ui;max-width:1250px;margin:auto;padding:24px;background:#fafafa}header{position:sticky;top:0;background:#fff;padding:12px;border-bottom:1px solid #ccc}#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}article{background:white;padding:12px;border:1px solid #ddd}img{display:block;max-width:220px;height:190px;object-fit:contain}button,input,select{font:inherit;padding:8px;margin:4px}small{display:block;overflow-wrap:anywhere}details{font-size:.85rem}#notice{color:#705000}</style>
<header><h1>Independent relevance review</h1><p>No model rankings are used in this worksheet. Unjudged is different from irrelevant.</p>
<label>Annotator ID <input id="annotator" autocomplete="off"></label><label>Query <select id="query"></select></label><button id="save">Download graded annotations CSV</button><p id="progress"></p></header>
<p>Grades:0 irrelevant or a clear mandatory-attribute conflict;1 weak/partial match;2 useful relevant result;3 clear match for all requested visible attributes. Leave uncertain facts unjudged and record why. Source attributes may help identify products but do not prove hidden material or properties.</p>
<p id="notice">This candidate pool is a seeded independent subset. It is not a complete relevance set, and unjudged products must not be called human negatives. Recall remains unavailable without complete judgements. Keep your own saved CSV; browser form state is not a research record until exported.</p><main id="cards"></main>
<script>const rows=__DATA__;const grades=new Map();const qs=[...new Map(rows.map(r=>[r.query_id,r.query_text])).entries()];const query=document.getElementById('query');const annotator=document.getElementById('annotator');const cards=document.getElementById('cards');
for(const [id,text] of qs){const option=document.createElement('option');option.value=id;option.textContent=id+' — '+text;query.append(option)}
function progress(){document.getElementById('progress').textContent=grades.size+' graded pairs / '+rows.length+' candidates; '+qs.length+' queries'}
function show(){cards.replaceChildren();for(const row of rows.filter(r=>r.query_id===query.value)){const key=row.query_id+'|'+row.product_id;const card=document.createElement('article');const title=document.createElement('h3');title.textContent=row.product_id;card.append(title);const img=document.createElement('img');img.src=row.image;img.alt='Source catalogue product';card.append(img);const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Source product attributes';details.append(summary);const text=document.createElement('p');text.textContent=row.attributes;details.append(text);card.append(details);const select=document.createElement('select');for(const [value,label] of [['','Unjudged'],['0','0 — irrelevant'],['1','1 — partial'],['2','2 — relevant'],['3','3 — strong match']]){const option=document.createElement('option');option.value=value;option.textContent=label;select.append(option)}select.value=grades.get(key)?.relevance??'';const notes=document.createElement('input');notes.placeholder='Judgement notes';notes.value=grades.get(key)?.notes??'';function update(){if(select.value===''){grades.delete(key)}else{grades.set(key,{...row,relevance:select.value,notes:notes.value})}progress()}select.onchange=update;notes.oninput=update;card.append(select,notes);cards.append(card)}progress()}
query.onchange=show;document.getElementById('save').onclick=()=>{if(!annotator.value.trim()){alert('Enter your annotator ID before exporting.');return}const fields=['query_id','product_id','relevance','annotator_id','annotation_date','notes'];const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const output=[fields.join(',')];for(const row of grades.values()){const item={...row,annotator_id:annotator.value.trim(),annotation_date:new Date().toISOString().slice(0,10)};output.push(fields.map(f=>quote(item[f])).join(','))}const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([output.join('\r\n')],{type:'text/csv;charset=utf-8'}));link.download='human_annotations.csv';link.click();URL.revokeObjectURL(link.href)};show();</script></html>'''
    output = target / "index.html"
    output.write_text(page.replace("__DATA__", data), encoding="utf-8")
    return {"html": str(output), "candidate_pairs": len(rows), "prepopulated_ratings": 0, "model_rankings_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(export(parser.parse_args().root), indent=2))
