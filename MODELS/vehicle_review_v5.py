"""Frozen blinded peer comparisons and family-level promotion gates."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from MODELS import vehicle_profiles as r


def export(run, target, approach="category_cluster"):
    target = Path(target)
    for grain in ("cohort", "model"):
        directory = target / grain
        directory.mkdir(exist_ok=True)
        source = Path(run) / grain
        queries = pd.read_csv(source / "evaluation_queries.csv")
        profiles = pd.read_parquet(source / "variant_profiles.parquet").set_index("profile_id")
        peers = pd.read_parquet(source / "profile_peers.parquet")
        rng = np.random.default_rng(42)
        rows, keys, cards = [], [], []
        for query in queries.itertuples():
            order = ["direct_specs", approach]
            if rng.integers(2):
                order.reverse()
            row = {"card_id": query.card_id, "query_profile_id": query.profile_id, "reviewer": "", "winner": ""}
            name = " / ".join(str(profiles.loc[query.profile_id,c]) for c in r.COHORT if pd.notna(profiles.loc[query.profile_id,c]))
            parts = [f'<section><h2>{html.escape(query.card_id)}: {html.escape(name)}</h2><p>{grain} discovery. Unknown electrification is unverified. Grade 0 unsuitable, 1 plausible, 2 strong.</p>']
            for side, method in zip(("A","B"), order):
                keys.append({"card_id": query.card_id, "side": side, "approach": method})
                selected = peers[peers.query_profile_id.eq(query.profile_id) & peers.approach.eq(method)].set_index("rank")
                parts.append(f'<h3>Slate {side}</h3>')
                for rank in range(1,6):
                    pid = selected.loc[rank,"peer_profile_id"] if rank in selected.index else ""
                    row[f"{side}{rank}_peer_id"], row[f"{side}{rank}_rating"] = pid, ""
                    if pid:
                        vehicle = profiles.loc[pid]
                        details = "; ".join(f"{c.replace('_',' ')}: {vehicle[c]}" for c in [*r.COHORT,"body_style","powertrain","seating","wheelbase","height","horsepower","wheel_system"] if pd.notna(vehicle[c]))
                        parts.append(f'<p>{rank}. {html.escape(details)} <select data-card="{query.card_id}" data-field="{side}{rank}_rating"><option></option><option>0</option><option>1</option><option>2</option></select></p>')
                    else:
                        parts.append(f'<p>{rank}. No supported alternative</p>')
            parts.append(f'<label>Preference <select data-card="{query.card_id}" data-field="winner"><option></option><option>A</option><option>B</option><option>tie</option><option>neither</option></select></label></section>')
            rows.append(row)
            cards.extend(parts)
        template = pd.DataFrame(rows)
        template.to_csv(directory / "ratings_template.csv", index=False)
        pd.DataFrame(keys, columns=["card_id","side","approach"]).to_csv(directory / "blind_key.csv", index=False)
        payload = json.dumps(rows).replace("</", "<\\/")
        script = """<script>
const rows=PAYLOAD; const namespace='peer-review-RUN-GRAIN';
let saved=JSON.parse(localStorage.getItem(namespace)||'{}');
document.querySelectorAll('select').forEach(el=>{el.value=saved[el.dataset.card]?.[el.dataset.field]||'';el.onchange=()=>{(saved[el.dataset.card]??={})[el.dataset.field]=el.value;localStorage.setItem(namespace,JSON.stringify(saved));};});
document.getElementById('export').onclick=()=>{const reviewer=document.getElementById('reviewer').value;const cols=Object.keys(rows[0]||{});const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';const out=[cols.map(esc).join(',')];rows.forEach(row=>{Object.assign(row,saved[row.card_id]||{},{reviewer});out.push(cols.map(c=>esc(row[c])).join(','));});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out.join('\\n')],{type:'text/csv'}));a.download='peer_ratings.csv';a.click();};
</script>""".replace("PAYLOAD", payload).replace("RUN", r.file_hash(source / "manifest.json")[:16]).replace("GRAIN", grain)
        (directory / "review_cards.html").write_text('<!doctype html><meta charset="utf-8"><title>Blinded peer review</title><style>body{font:16px system-ui;margin:2rem;max-width:1100px}section{border-bottom:1px solid #ccc;padding:1rem}select,input,button{padding:.4rem}</style><h1>Blinded peer review</h1><p>Keep blind_key.csv separate until judgments are complete. The same alternative must receive the same grade within a card.</p><label>Reviewer <input id="reviewer"></label><button id="export">Export ratings CSV</button>'+''.join(cards)+script, encoding="utf-8")
    return {"approach": approach, "run_directory": str(Path(run).resolve()), "run_manifest_sha256": r.file_hash(Path(run) / "manifest.json")}


def evaluate(review, ratings_directory, run, smoke=False):
    results = {}
    for grain in ("cohort","model"):
        directory = Path(review) / grain
        expected = pd.read_csv(directory / "ratings_template.csv", dtype=str, keep_default_na=False)
        actual = pd.read_csv(Path(ratings_directory) / grain / "ratings.csv", dtype=str, keep_default_na=False)
        identity = ["card_id","query_profile_id", *[f"{side}{rank}_peer_id" for side in ("A","B") for rank in range(1,6)]]
        if set(actual.columns) != set(expected.columns) or actual.card_id.duplicated().any() or set(actual.card_id) != set(expected.card_id):
            raise ValueError("Review identities or columns changed")
        actual = actual.set_index("card_id").loc[expected.card_id].reset_index()
        if not actual[identity].equals(expected[identity]) or not actual.reviewer.str.strip().ne("").all() or not actual.winner.isin(["A","B","tie","neither"]).all():
            raise ValueError("Frozen identities, reviewer or preference invalid")
        key = pd.read_csv(directory / "blind_key.csv")
        values, family_seen = [], set()
        p = pd.read_parquet(Path(run)/grain/"variant_profiles.parquet").set_index("profile_id")
        for row in actual.to_dict("records"):
            family = p.loc[row["query_profile_id"],"family_block"]
            if family in family_seen:
                raise ValueError("Queries must represent distinct families")
            family_seen.add(family)
            score, grades_seen = {}, {}
            for side in ("A","B"):
                grades=[]
                for rank in range(1,6):
                    pid, grade = row[f"{side}{rank}_peer_id"], row[f"{side}{rank}_rating"]
                    if pid:
                        if grade not in ("0","1","2") or pid in grades_seen and grades_seen[pid] != grade:
                            raise ValueError("Returned alternatives require consistent grades 0/1/2")
                        grades_seen[pid] = grade
                        grades.append(int(grade)>=1)
                    elif grade:
                        raise ValueError("Unavailable slots cannot be graded")
                    else:
                        grades.append(False)
                method = key.loc[key.card_id.eq(row["card_id"]) & key.side.eq(side),"approach"].item()
                score[method] = float(np.mean(grades))
            enhanced = next(m for m in score if m != "direct_specs")
            values.append(score[enhanced]-score["direct_specs"])
        intervals=[]
        for seed in (42,43):
            rng=np.random.default_rng(seed)
            draws=np.asarray(values)[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
            intervals.append(np.quantile(draws,[.025,.975]).tolist())
        statuses=pd.read_parquet(Path(run)/grain/"query_status.parquet").pivot(index="query_profile_id",columns="approach",values="slate_size")
        no_loss=bool(statuses[enhanced].ge(statuses.direct_specs).all())
        accepted=len(values)>=100 and not smoke and no_loss and all(ci[0]>0 for ci in intervals)
        results[grain]={"families":len(values),"useful_at_5_gain":float(np.mean(values)),"confidence_intervals":intervals,"no_coverage_loss":no_loss,"selected_approach":enhanced if accepted else "direct_specs"}
    return results
