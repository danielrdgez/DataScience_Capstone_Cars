"""Bounded public-reference acquisition; unsupported scope stays unresolved."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from MODELS import vehicle_profiles as r
from MODELS import vehicle_taxonomy_v2 as v2

HEADERS = {"User-Agent": "VehicleTaxonomyResearch/2.0 (educational vehicle classification source verification)"}
CLASSIFICATIONS = [
    ("subcompact car", "size_segment", "subcompact"), ("compact car", "size_segment", "compact"),
    ("mid-size car", "size_segment", "midsize"), ("full-size car", "size_segment", "large"),
    ("subcompact crossover suv", "size_segment", "small_suv"), ("compact crossover suv", "size_segment", "small_suv"),
    ("mid-size crossover suv", "size_segment", "midsize_suv"), ("mid-size suv", "size_segment", "midsize_suv"),
    ("full-size suv", "size_segment", "large_suv"), ("full-size pickup truck", "size_segment", "full_size_pickup"),
    ("compact pickup truck", "size_segment", "small_pickup"), ("sports car", "performance_character", "sports_car"),
    ("muscle car", "performance_character", "muscle_car"), ("grand tourer", "performance_character", "grand_tourer"),
    ("supercar", "performance_character", "supercar"), ("hypercar", "performance_character", "hypercar"),
    ("economy car", "market_positioning", "economy"), ("luxury", "market_positioning", "luxury"),
    ("compact executive car", "market_positioning", "luxury")]


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def acquire_one(task):
    make, model, start, end, page_title = task
    result = {"make": make, "model": model, "observed_year_start": start, "observed_year_end": end,
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "status": "unavailable_classification", "claims": []}
    try:
        url = "https://en.wikipedia.org/wiki/" + page_title.replace(" ", "_")
        response = requests.get(url, headers=HEADERS, timeout=25)
        response.raise_for_status()
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        heading = soup.select_one("h1")
        title = heading.get_text(" ",strip=True) if heading else page_title
        result["title"] = title
        result["url"] = url
        if normalized(title) != normalized(make + model):
            result["status"] = "unmatched_identity"
            return result
        table = soup.select_one("table.infobox")
        if table is None:
            return result
        rows = {}
        for row in table.select("tr"):
            heading, value = row.find("th"), row.find("td")
            if heading and value:
                for sup in value.select("sup.reference"):
                    sup.decompose()
                rows[" ".join(heading.get_text(" ",strip=True).lower().split())] = value.get_text(" ",strip=True)
        classification = rows.get("class", "")
        years = rows.get("model years", "")
        result.update(url=url, class_passage=classification[:500], model_year_passage=years[:500],
            manufacturer_links=sorted({a.get("href") for a in table.select("a.external") if a.get("href", "").startswith("https://")}))
        # Production years alone are not model-year applicability.
        ranges = re.findall(r"(19\d{2}|20\d{2})\s*[\u2013\u2014-]\s*(present|19\d{2}|20\d{2})", years.lower())
        if len(ranges) != 1:
            result["status"] = "ambiguous_scope"
            return result
        first, last = ranges[0]
        first, last = int(first), 2026 if last == "present" else int(last)
        # Multiple historical classes must be scoped separately, never flattened.
        if re.search(r"\((?:[^)]*\d{4})", classification) or len(classification) > 180:
            result["status"] = "ambiguous_scope"
            return result
        if "compact executive car" in classification.lower() and "/" in classification:
            result["status"] = "ambiguous_scope"
            return result
        text = classification.lower().replace("\u2011", "-")
        claims = [{"dimension":dimension,"label":label} for phrase,dimension,label in CLASSIFICATIONS if re.search(r"(?<![a-z])"+re.escape(phrase)+r"(?![a-z])",text)]
        # A compact executive term is a documented terminology crosswalk, not make inference.
        result.update(year_start=max(start,first), year_end=min(end,last), claims=claims,
            status="source_supported" if claims and max(start,first)<=min(end,last) else "unavailable_classification")
        if "compact executive car" in text:
            result["crosswalk"] = "compact_executive_to_luxury"
            result["supporting_urls"] = ["https://en.wikipedia.org/wiki/Compact_executive_car"]
        return result
    except (requests.RequestException, ValueError, KeyError) as error:
        result.update(status="reference_unavailable",error=str(error)[:200])
        return result


def acquire(prepared, output, previous_batches=()):
    profiles = pd.read_parquet(Path(prepared)/"variant_profiles.parquet")
    names = pd.read_parquet(Path(prepared)/"identity_comparison.parquet")
    profiles["market_family"] = profiles.Model
    previous = set()
    for directory in previous_batches:
        previous.update((x["make"],x["model"]) for x in r.read_json(Path(directory)/"reference_results.json"))
    if previous:
        profiles = profiles[~pd.MultiIndex.from_frame(profiles[["Make","market_family"]]).isin(previous)].copy()
    batch = v2.evidence_batch(profiles)
    # Keep all existing rare performance families in the first batch.
    registry = r.read_json(v2.ASSETS/"evidence.v1.json")
    rare = {(x["make"],x["model"]) for x in registry["sources"] if any(c["dimension"]=="performance_character" for c in x["claims"])}
    selected = list(batch[["Make","market_family"]].itertuples(index=False,name=None))
    extras = [pair for pair in sorted(rare) if pair not in selected and pair not in previous]
    if previous:
        # Half volume priority, half round-robin diversity from the remaining inventory.
        selected = selected[:50] + selected[100:150]
    else:
        selected = selected[:max(0, 200-len(extras))] + extras
    tasks = []
    for make,model in selected:
        subset = profiles[profiles.Make.eq(make)&profiles.Model.eq(model)]
        if len(subset):
            alternatives=names[names.Make.eq(make)&names.Model.eq(model)].groupby(["make_name","model_name"]).vin_count.sum().sort_values(ascending=False)
            page_title=" ".join(alternatives.index[0]) if len(alternatives) else make.title()+" "+model.title()
            tasks.append((make,model,int(pd.to_numeric(subset.ModelYear).min()),int(pd.to_numeric(subset.ModelYear).max()),page_title))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(acquire_one,tasks),1):
            results.append(result)
            if i%20==0:
                print("References checked:",i,"/",len(tasks),flush=True)
    r.write_json(output/"reference_results.json",results)
    pd.DataFrame([{"make":x[0],"model":x[1],"year_start":x[2],"year_end":x[3],"page_title":x[4],"priority":i+1} for i,x in enumerate(tasks)]).to_csv(output/"selection.csv",index=False)
    (output/"execution_reference_acquisition.py").write_bytes(Path(__file__).read_bytes())
    r.write_json(output/"receipt.json",{"requested_families":len(tasks),"results":len(results),"code_sha256":r.file_hash(__file__),
        "prepared_manifest_sha256":r.file_hash(Path(prepared)/"manifest.json"),"previous_batches":[str(p) for p in previous_batches],"method":"targeted_exact_title_infobox_and_explicit_model_year_scope",
        "manufacturer_preference":"manually targeted manufacturer references supplement conservative Wikipedia discovery"})
    print(pd.Series([x["status"] for x in results]).value_counts().to_string(),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--previous-batch",type=Path,action="append",default=[],help="Exclude already attempted identities; use a 100-family follow-up batch")
    args=parser.parse_args()
    acquire(args.prepared,args.output,args.previous_batch)
