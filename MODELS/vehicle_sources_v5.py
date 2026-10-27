"""Resumable reference acquisition; a failed request is never classification evidence."""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import urlencode
from urllib.parse import quote

import pandas as pd
import requests

from MODELS import vehicle_profiles as research
from MODELS import vehicle_similarity_v4 as v4
from MODELS import vehicle_taxonomy_sources as vocabulary

PARSER_VERSION = "5.0.0"
API = "https://en.wikipedia.org/w/api.php"
EPA_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"
CLASSIFICATION_REFERENCES = [
    "https://en.wikipedia.org/wiki/Car_classification",
    "https://en.wikipedia.org/wiki/Vehicle_size_class#NHTSA",
    "https://www.nhtsa.gov/ratings",
    "https://www.fueleconomy.gov/feg/byclass/byepaclassnf.shtml",
    "https://www.fueleconomy.gov/feg/info.shtml#sizeclasses",
]
TRANSIENT = {408, 425, 429, 500, 502, 503, 504}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    v4.atomic_json(path, value)


class DeferredReference(RuntimeError):
    pass


class ReferenceFailure(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(status)


class ReferenceClient:
    """One serial host queue, bounded retries, durable request history and successful cache."""

    def __init__(self, directory, interval=1., attempts=5, contact="https://github.com/danielrdgez/DataScience_Capstone_Cars"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"VehicleTaxonomyResearch/5.0 ({contact})", "Accept-Encoding": "gzip"})
        self.interval, self.attempts, self.last_request = interval, attempts, 0.
        self.cache_hits = self.requests = 0

    def get(self, url, *, binary=False):
        key = research.digest(url)
        file = self.directory / (key + ".json")
        state = json.loads(file.read_text()) if file.exists() else {"requested_url": url, "attempts": []}
        if state.get("status") == "success":
            self.cache_hits += 1
            return state
        if state.get("next_retry_at", 0) > time.time():
            raise DeferredReference("Retry deferred until " + str(state["next_retry_at"]))
        for attempt in range(self.attempts):
            time.sleep(max(0., self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            self.requests += 1
            response = None
            try:
                response = self.session.get(url, timeout=(10, 35))
                api_error = None
                if not binary and "json" in response.headers.get("Content-Type", ""):
                    api_error = response.json().get("error", {}).get("code")
                transient = response.status_code in TRANSIENT or api_error in ("maxlag", "ratelimited")
                state["attempts"].append({"at": now(), "http_status": response.status_code, "api_error": api_error})
                if not transient:
                    response.raise_for_status()
                    if api_error:
                        raise ValueError("MediaWiki error: " + api_error)
                    state.update(status="success", canonical_url=response.url, retrieved_at=now(),
                                 content_sha256=hashlib.sha256(response.content).hexdigest(), next_retry_at=0)
                    if binary:
                        payload = self.directory / (key + ".zip")
                        payload.write_bytes(response.content)
                        state["payload"] = str(payload.resolve())
                    else:
                        response.encoding = "utf-8"
                        state["html"] = response.text
                    save(file, state)
                    return state
            except (requests.RequestException, ValueError) as error:
                state["attempts"].append({"at": now(), "error": str(error)[:300]})
                if response is not None and response.status_code not in TRANSIENT:
                    state.update(status="missing_page" if response.status_code == 404 else "request_rejected", next_retry_at=0)
                    save(file, state)
                    return state
            delay = 5 * 2 ** attempt + random.uniform(0, 1)
            if response is not None and response.headers.get("Retry-After"):
                value = response.headers["Retry-After"]
                try:
                    delay = max(delay, float(value))
                except ValueError:
                    delay = max(delay, parsedate_to_datetime(value).timestamp() - time.time())
            state.update(status="deferred", next_retry_at=time.time() + delay)
            save(file, state)
            # Long cooldowns belong in the persisted queue, not a sleeping worker.
            if delay > 60 or attempt + 1 == self.attempts:
                raise DeferredReference("Transient request remains queued: " + url)
            time.sleep(delay)
        raise DeferredReference(url)

    def api(self, **parameters):
        url = API + "?" + urlencode({"action": "query", "format": "json", "formatversion": 2, "maxlag": 5, **parameters})
        document = self.get(url)
        if document["status"] != "success":
            raise ReferenceFailure(document["status"])
        return json.loads(document["html"]), document


def parse_page(document, task):
    """Reuse exact-scope parsing, adding title aliases and separate generation infoboxes."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(document["html"], "html.parser")
    heading = soup.select_one("h1")
    title = heading.get_text(" ", strip=True) if heading else ""
    expected = {vocabulary.normalized(task["make"] + task["model"]), *task.get("accepted_titles", [])}
    base = vocabulary.normalized(re.sub(r"\s*\([^)]*\)\s*$", "", title))
    if base not in expected:
        return [], "unmatched_identity"
    entries, reasons = [], []
    for table in soup.select("table.infobox"):
        for sup in table.select("sup.reference"):
            sup.decompose()
        single = {**document, "url": document.get("url", document.get("canonical_url")),
                  "html": "<h1>" + task["make"] + " " + task["model"] + "</h1>" + str(table)}
        found, reason = v4.parse_document(single, task)
        for entry in found:
            entry.update(parser_version=PARSER_VERSION, page_id=document.get("page_id"), revision_id=document.get("revision_id"))
        entries.extend(found)
        reasons.append(reason)
    return entries, "source_supported" if entries else next((r for r in ["ambiguous_scope", "out_of_scope", "unavailable_classification"] if r in reasons), "unavailable_classification")


def acquire(task, client):
    result = {"make": task["make"], "model": task["model"], "entries": [], "attempts": [], "parser_version": PARSER_VERSION}
    try:
        search, _ = client.api(list="search", srsearch=task["make"] + " " + task["model"], srlimit=5)
        titles = [h["title"] for h in search.get("query", {}).get("search", [])]
        any_candidates=bool(titles)
        expected = {vocabulary.normalized(task["make"] + task["model"]), *task.get("accepted_titles", [])}
        titles = [title for title in titles if vocabulary.normalized(re.sub(r"\s*\([^)]*\)\s*$", "", title)) in expected]
        if not titles:
            result["attempts"].append({"status": "unmatched_identity" if any_candidates else "missing_page"})
        for title in titles:
            metadata, _ = client.api(titles=title, redirects=1, prop="info|revisions", inprop="url", rvprop="ids", rvlimit=1)
            for page in metadata.get("query", {}).get("pages", []):
                if page.get("missing"):
                    result["attempts"].append({"status": "missing_page", "title": title})
                    continue
                url = API + "?" + urlencode({"action": "parse", "format": "json", "pageid": page["pageid"], "prop": "text", "maxlag": 5})
                cached = client.get(url)
                if cached["status"] != "success":
                    result["attempts"].append({"status": cached["status"], "url": url})
                    continue
                parsed = json.loads(cached["html"]).get("parse", {})
                html = parsed.get("text", {}).get("*", "")
                document = {**cached, "html": "<h1>" + page["title"] + "</h1>" + html,
                            "url": page["fullurl"], "page_id": page["pageid"],
                            "revision_id": page.get("revisions", [{}])[0].get("revid")}
                entries, reason = parse_page(document, task)
                result["entries"].extend(entries)
                result["attempts"].append({"status": reason, "url": page["fullurl"], "revision_id": document["revision_id"]})
        reasons = {a["status"] for a in result["attempts"]}
        result["status"] = "source_supported" if result["entries"] else next((r for r in
            ["ambiguous_scope", "out_of_scope", "unavailable_classification", "unmatched_identity", "request_rejected", "missing_page"] if r in reasons), "missing_page")
    except ReferenceFailure as error:
        result.update(status=error.status, error=str(error))
    except (DeferredReference, requests.RequestException, ValueError) as error:
        result.update(status="deferred", error=str(error)[:300])
    return result


def epa_index(table):
    index = {}
    normalize = vocabulary.normalized
    for idx, row in table.iterrows():
        for model in {str(row.model), str(row.get("baseModel", row.model))}:
            index.setdefault((int(row.year),normalize(row.make),normalize(model)),set()).add(idx)
    return index


def nhtsa_class(profile):
    """Derived NCAP class; curb-weight evidence never establishes a safety rating."""
    body=profile.get("body_family")
    if body in ("suv_crossover","pickup","van_minivan"):
        return {"suv_crossover":"SUV","pickup":"PU","van_minivan":"VAN"}[body]
    if body!="passenger_car" or profile.get("curb_weight_lb_n",0)<5 or profile.get("curb_weight_lb_n",0)/profile["vin_count"]<.9:
        return None
    low,high=profile.get("curb_weight_lb_min"),profile.get("curb_weight_lb_max")
    if pd.isna(low) or pd.isna(high) or low<1500:
        return None
    bands=[(1500,2000,"PC/Mi"),(2000,2500,"PC/L"),(2500,3000,"PC/C"),(3000,3500,"PC/Me"),(3500,float("inf"),"PC/H")]
    return next((name for start,end,name in bands if low>=start and high<end),None)


def epa_candidates(profile, table, index=None):
    """Only documented aliases and compatible variants; all remaining records must agree."""
    normalize = vocabulary.normalized
    names = {normalize(profile["Model"]), normalize(profile.get("market_family", profile["Model"]))}
    if index is None:
        index=epa_index(table)
    indices=set()
    for model in names:
        indices.update(index.get((int(profile["ModelYear"]),normalize(profile["Make"]),model),set()))
    rows=table.loc[sorted(indices)]
    if not len(rows):
        return rows
    drive_map = {"Front-Wheel Drive": "FWD", "Rear-Wheel Drive": "RWD", "All-Wheel Drive": "AWD", "4-Wheel Drive": "4WD", "2-Wheel Drive": "2WD", "4-Wheel or All-Wheel Drive":"ALL_WHEEL"}
    detail = profile.get("wheel_system")
    if pd.notna(detail) and "drive" in rows:
        drives = rows.drive.map(drive_map)
        compatible = drives.eq(detail) | (drives.eq("2WD") & (detail in ("FWD", "RWD", "2WD")))
        if detail in ("AWD", "4WD", "ALL_WHEEL"):
            compatible |= drives.isin(["AWD", "4WD", "ALL_WHEEL"])
        rows = rows[compatible | drives.isna()]
    body=profile.get("body_family")
    if body in ("passenger_car","suv_crossover","pickup","van_minivan"):
        classes=rows.VClass.fillna("")
        pattern={"passenger_car":"Cars|Wagons|Two Seaters|Two-Seaters", "suv_crossover":"Sport Utility", "pickup":"Pickup", "van_minivan":"Vans|Minivans"}[body]
        rows=rows[classes.str.contains(pattern,case=False,regex=True)]
    if "fuelType1" in rows:
        electric=rows.fuelType1.fillna("").str.lower().eq("electricity")
        power=profile.get("powertrain")
        if power=="battery_electric":
            rows=rows[electric]
        elif power in ("combustion","combustion_unverified","hybrid_unspecified"):
            rows=rows[~electric]
    if profile.get("powertrain")=="hybrid_unspecified" and "atvType" in rows:
        rows=rows[rows.atvType.isin(["Hybrid","Plug-in Hybrid"])]
    if profile.get("powertrain") in ("hybrid","plug_in_hybrid") and "atvType" in rows:
        rows=rows[rows.atvType.eq("Hybrid" if profile["powertrain"]=="hybrid" else "Plug-in Hybrid")]
    for field, reference, divisor, tolerance in [("engine_displacement", "displ", 1000., .15), ("engine_cylinders", "cylinders", 1., 0.)]:
        value = profile.get(field)
        if pd.notna(value) and reference in rows:
            values = pd.to_numeric(rows[reference], errors="coerce")
            rows = rows[values.isna() | values.sub(value / divisor).abs().le(tolerance)]
    transmission = str(profile.get("transmission", "")).upper()
    if transmission in ("A", "M") and "trany" in rows:
        known = rows.trany.fillna("").str.lower()
        rows = rows[known.eq("") | known.str.contains("manual" if transmission == "M" else "automatic", regex=False)]
    return rows


def epa_claims(profile, table, index=None):
    rows = epa_candidates(profile, table, index)
    result = {"candidate_ids": rows.id.astype(str).tolist(), "reason": "missing_reference" if rows.empty else "ambiguous_variant"}
    if rows.empty:
        return result
    classes = rows.VClass.dropna().unique()
    if len(classes) == 1 and rows.VClass.notna().all():
        result.update(epa_class=str(classes[0]), reason="source_supported")
    # Empty atvType is not evidence that a vehicle is non-electrified.
    if "atvType" in rows:
        power = rows.atvType.map({"EV": "battery_electric", "Hybrid": "hybrid", "Plug-in Hybrid": "plug_in_hybrid"})
        if power.notna().all() and power.nunique() == 1:
            result["powertrain"] = power.iloc[0]
    return result


def cached_pages(directory):
    """Index successful cached HTML/API documents for offline scope re-evaluation."""
    index={}
    for file in Path(directory).glob("*.json"):
        document=json.loads(file.read_text())
        text=document.get("html","")
        if not text or document.get("status","success")!="success":
            continue
        if text.lstrip().startswith("{"):
            payload=json.loads(text).get("parse",{})
            title=payload.get("title","")
            text=payload.get("text",{}).get("*","")
            url="https://en.wikipedia.org/wiki/"+quote(title.replace(" ","_"))
            document={**document,"url":url,"page_id":payload.get("pageid"),"revision_id":payload.get("revid")}
        else:
            heading=re.search(r"<h1\b[^>]*>(.*?)</h1>",text,re.I|re.S)
            title=html.unescape(re.sub(r"<[^>]+>","",heading.group(1))) if heading else ""
            document={**document,"url":document.get("url",document.get("canonical_url",document.get("requested_url")))}
        if not title or "infobox" not in text:
            continue
        key=vocabulary.normalized(re.sub(r"\s*\([^)]*\)\s*$","",title))
        index.setdefault(key,[]).append({**document,"html":"<h1>"+title+"</h1>"+text})
    return index
