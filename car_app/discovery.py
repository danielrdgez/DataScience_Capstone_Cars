"""Saved v6 discovery charts and deterministic evidence-backed shortlists."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from MODELS import discovery_artifacts as a
from MODELS.vehicle_similarity_v6 import NUMERIC
from car_app.artifacts import stat_key
from car_app.data import COHORT

COMPONENTS = ["powertrain", "brakes", "steering", "electrical", "airbags", "fuel", "structure", "visibility"]
SPEC_LABELS = {"seating": "Seating (seats)", "wheelbase": "Wheelbase (in)", "length": "Length (in)",
               "width": "Width (in)", "height": "Height (in)", "front_legroom": "Front legroom (in)",
               "back_legroom": "Back legroom (in)", "horsepower": "Horsepower (hp)",
               "torque_nm": "Torque (Nm)", "engine_displacement": "Displacement (cc)"}
COMPARABLE_RATINGS = ["OverallSideCrashRating", "RolloverRating", "SidePoleCrashRating"]
RATING_LABELS = {"OverallSideCrashRating": "Side crash (stars)", "RolloverRating": "Rollover (stars)",
                 "SidePoleCrashRating": "Side pole (stars)"}


def checked_frame(owner, relative):
    owner = Path(owner)
    expected = a.read_json(owner / "manifest.json")["artifacts"].get(relative)
    path = owner / relative
    if not expected or a.file_hash(path) != expected:
        raise ValueError("Saved discovery table changed: " + str(path))
    return pd.read_parquet(path)


@st.cache_data(max_entries=128, show_spinner=False)
def check_saved_file(path, signature, expected):
    if not expected or a.file_hash(path) != expected:
        raise ValueError("Saved discovery artifact changed: " + str(path))
    return True


@st.cache_data(max_entries=16, show_spinner=False)
def family_records(text_directory, period, family_ids, signature):
    owner = Path(text_directory)
    relative = period + "/records.parquet"
    path = owner / relative
    expected = a.read_json(owner / "manifest.json")["artifacts"].get(relative)
    check_saved_file(str(path), signature, expected)
    return pl.scan_parquet(path).filter(pl.col("family_id").is_in(list(family_ids))).collect().to_pandas()


@st.cache_resource(max_entries=4, show_spinner=False)
def load_catalog(run, signature, expected_hash, source_identity, source_directory, source_signature,
                 validation_directory, validation_signature, grain="model"):
    path = a.ROOT / "MODEL_OUTPUT" / run
    if a.file_hash(path / "manifest.json") != expected_hash:
        raise ValueError("Pinned discovery manifest changed")
    receipt = a.verify(path, full=False)
    if receipt.get("reduced_budget") or receipt["stage"] != "attach-evidence":
        raise ValueError("Only completed full discovery bundles can be displayed")
    validation_path = Path(validation_directory)
    validation = a.verify(validation_path, full=False)
    expected_run = {"directory": str(path.resolve()), "sha256": expected_hash}
    if validation.get("stage") != "validate" or validation.get("reduced_budget") or not validation.get("passed") or validation.get("run") != expected_run:
        raise ValueError("Discovery validation does not match the selected full bundle")
    check_saved_file(str(validation_path / "validation.json"), validation_signature,
                     validation["artifacts"].get("validation.json"))
    report = a.read_json(validation_path / "validation.json")
    if not report.get("passed") or report.get("run") != expected_run:
        raise ValueError("Discovery validation report does not match the selected full bundle")
    if a.source_hashes(source_directory) != receipt["source_hashes"]:
        raise ValueError("Discovery sources differ from the current snapshot; rebuild the bundle")
    catalog = Catalog(path, receipt, grain)
    catalog.validation = report
    return catalog


class Catalog:
    def __init__(self, path, receipt, grain):
        self.path, self.receipt, self.grain = Path(path), receipt, grain
        self.specs, self.text = Path(receipt["specs"]), Path(receipt["text"])
        self.profiles = checked_frame(self.specs, f"{grain}/profiles.parquet")
        self.members = checked_frame(self.specs, f"{grain}/memberships.parquet")
        self.assignments = checked_frame(self.specs, f"{grain}/assignments.parquet")
        self.safety = checked_frame(self.path, "safety.parquet")

    def scope(self, source, filters):
        from car_app.recommendation import matching_profile_statistics
        path = self.specs / self.grain / "memberships.parquet"
        counts = matching_profile_statistics(source.identity, filters, str(path), stat_key(path), _source=source)
        return self.profiles.merge(counts, on="profile_id", validate="1:1")

    def context(self, profile):
        category = profile["category"]
        return a.digest(category)[:12] + "-fold-" + str(int(profile["fold"]))

    def components(self, period):
        return checked_frame(self.text, period + "/family_components.parquet")

    def specification_peers(self, reference):
        from MODELS import vehicle_similarity_v3 as v3

        q = self.profiles.loc[self.profiles.profile_id.eq(reference)]
        if q.empty:
            return pd.DataFrame(columns=["profile_id", "specification_distance"])
        row = q.iloc[0]
        context = self.context(row)
        path = self.specs / self.grain / "contexts" / context
        if not path.exists():
            return pd.DataFrame(columns=["profile_id", "specification_distance"])
        model = joblib.load(path / "retrieval.joblib")
        support = self.assignments.loc[self.assignments.context.eq(context)].set_index("profile_id")
        if reference not in support.index or not bool(support.loc[reference, "eligible"]) or not model["metric"]:
            return pd.DataFrame(columns=["profile_id", "specification_distance"])
        candidates = self.profiles.loc[self.profiles.category.eq(row.category) & self.profiles.family_block.ne(row.family_block)]
        candidates = candidates.loc[candidates.profile_id.isin(support.loc[support.eligible, :].index)]
        candidates = candidates.loc[(pd.to_numeric(candidates.ModelYear) - int(row.ModelYear)).abs().le(10)]
        if candidates.empty:
            return pd.DataFrame(columns=["profile_id", "specification_distance"])
        distance = v3.distances(q, candidates, model["metric"])[0]
        return pd.DataFrame({"profile_id": candidates.profile_id.to_numpy(), "specification_distance": distance}).replace([np.inf, -np.inf], np.nan).dropna()

    def similar(self, reference, limit=5, candidate_ids=None):
        """Use one query-held-out fit for ranking and every displayed cluster label."""
        row = self.profiles.set_index("profile_id").loc[reference]
        context = self.context(row)
        peers = self.specification_peers(reference).merge(self.profiles, on="profile_id", validate="1:1")
        if candidate_ids is not None:
            peers = peers.loc[peers.profile_id.isin(candidate_ids)]
        peers = peers.sort_values(["specification_distance", "profile_id"]).drop_duplicates("family_block").head(limit)
        labels = self.assignments.loc[self.assignments.context.eq(context), ["profile_id", "cluster", "description", "status"]]
        peers = peers.merge(labels, on="profile_id", how="left", validate="1:1")
        peers.insert(0, "rank", np.arange(1, len(peers) + 1))
        query_labels = labels.loc[labels.profile_id.eq(reference)]
        query_cluster = int(query_labels.cluster.iloc[0]) if not query_labels.empty else -1
        peers["shared_cluster"] = peers.cluster.map(lambda value: bool(value == query_cluster) if pd.notna(value) and value >= 0 and query_cluster >= 0 else None)
        return peers, context


def active_catalog(source, grain="model"):
    from car_app.model_registry import registry

    config = registry()
    run = config.get("discovery_run")
    if not run:
        return None
    validation = config.get("discovery_validation")
    if not validation:
        raise ValueError("The selected discovery bundle has no validation receipt")
    validation_path = Path(validation)
    validation_signature = tuple(stat_key(validation_path / name) for name in ("manifest.json", "validation.json"))
    signature = tuple((name, stat_key(source.directory / (name + ".parquet"))) for name in a.TABLES)
    return load_catalog(run, stat_key(a.ROOT / "MODEL_OUTPUT" / run / "manifest.json"), config.get("discovery_manifest_sha256"),
                        source.identity, str(source.directory), signature, str(validation_path), validation_signature, grain)


def discovery_caption(catalog):
    st.caption(f"Discovery v6 · validated full run {catalog.path.parent.name} · {catalog.grain} profiles · completed {catalog.receipt['completed_at'][:10]}")


@st.cache_data(max_entries=4, show_spinner=False)
def saved_peer_review(directory, signature, bundle_hash):
    path = Path(directory)
    receipt = a.verify(path, full=False)
    if receipt.get("stage") != "compare-peers" or receipt.get("reduced_budget") or receipt.get("run", {}).get("sha256") != bundle_hash:
        raise ValueError("Peer comparison does not match the active full discovery bundle")
    for name in ("coverage.csv", "peer_review.jsonl"):
        if a.file_hash(path / name) != receipt["artifacts"].get(name):
            raise ValueError("Saved peer comparison changed: " + name)
    return pd.read_csv(path / "coverage.csv"), pd.read_json(path / "peer_review.jsonl", lines=True)


def peer_review_panel(catalog, key="v6_peer_review"):
    with st.expander("Experimental peer comparison"):
        st.caption("Saved review samples compare direct, cluster-assisted and text-assisted lists. Text-assisted lists have coverage-matched direct baselines. Quality remains unmeasured; these experiments do not change the default ranking.")
        if not st.checkbox("Show saved peer comparison", key=key + "_show"):
            return
        path = catalog.path.parent / "peer-comparison"
        if not (path / "manifest.json").exists():
            st.info("No saved peer comparison accompanies this bundle.")
            return
        signature = tuple(stat_key(path / name) for name in ("manifest.json", "coverage.csv", "peer_review.jsonl"))
        coverage, peers = saved_peer_review(str(path), signature, a.file_hash(catalog.path / "manifest.json"))
        grain = st.selectbox("Peer review population", ["model", "cohort"], key=key + "_grain")
        local = peers.loc[peers.grain.eq(grain)]
        profiles = catalog.profiles if catalog.grain == grain else checked_frame(catalog.specs, grain + "/profiles.parquet")
        names = profile_names(profiles)
        query = st.selectbox("Reviewed reference", sorted(local.query_profile_id.unique()), format_func=names.get, key=key + "_query")
        selected = local.loc[local.query_profile_id.eq(query)].copy()
        selected["peer"] = selected.peer_profile_id.map(names)
        st.dataframe(selected[["approach", "rank", "peer", "distance", "context"]], hide_index=True, width="stretch",
                     alt="Experimental v6 peer lists for the selected reference and their shared fitted context")
        st.dataframe(coverage.loc[coverage.grain.eq(grain) & coverage.query_profile_id.eq(query)], hide_index=True, width="stretch",
                     alt="Candidate support and returned results for each experimental approach")
        st.caption("Scores have approach-specific definitions and should be compared within each list.")
        st.download_button("Download peer review sample", (path / "peer_review.jsonl").read_text(encoding="utf-8"),
                           "peer_review_v6.jsonl", mime="application/x-ndjson", key=key + "_download")


def with_listing_clusters(frame, catalog):
    if frame.is_empty():
        return frame
    members = catalog.members.loc[catalog.members.vin.isin(frame["vin"].to_list())]
    own = catalog.assignments.loc[catalog.assignments.profile_id.isin(members.profile_id)].merge(
        catalog.profiles[["profile_id", "fold"]], on="profile_id", suffixes=("", "_profile"), validate="m:1")
    own = own.loc[own.fold.eq(own.fold_profile), ["profile_id", "description", "status", "context"]].rename(
        columns={"description": "cluster_description", "status": "cluster_status", "context": "cluster_context"})
    labels = members.merge(own, on="profile_id", how="left", validate="m:1")
    return frame.drop("profile_id", strict=False).join(pl.from_pandas(labels), on="vin", how="left", validate="1:1")


def similarity_dashboard(source, filters, key="v6_similarity"):
    st.header("Similar vehicles to the dashboard selection")
    grain = st.segmented_control("Similarity population", ["model", "cohort"], default="model", key=key + "_grain",
                                 format_func=lambda value: "Model level" if value == "model" else "Trim cohorts") or "model"
    catalog = active_catalog(source, grain)
    discovery_caption(catalog)
    st.caption("Dashboard filters select the reference VINs. Direct specification distance ranks the broader catalog within the same body category and ±10 model years; the reference Make/Model family is excluded. Buyer usefulness remains unmeasured.")
    choices = catalog.scope(source, filters)
    choices = choices.loc[choices.identity_complete].copy()
    if choices.empty:
        st.session_state.pop(key + "_reference", None)
        st.info("No complete VIN-supported reference profiles match the dashboard filters.")
        return
    scope = str(catalog.path) + grain + filters.identity()
    if st.session_state.get(key + "_scope") != scope:
        st.session_state.pop(key + "_reference", None)
        st.session_state[key + "_scope"] = scope
    names = profile_names(choices)
    reference = st.selectbox("Reference vehicle", list(names), index=0 if filters.active() else None,
                             format_func=names.get, key=key + "_reference")
    if reference is None:
        st.info("Select a reference vehicle to show v6 alternatives.")
        return
    reference_row = choices.loc[choices.profile_id.eq(reference)]
    with st.container(horizontal=True):
        st.metric("Reference listings", f"{int(reference_row.matching_listings.iloc[0]):,}")
        st.metric("Reference median advertised price", f"${reference_row.filtered_median_price.iloc[0]:,.0f}")
    peers, context = catalog.similar(reference)
    labels = catalog.assignments.loc[catalog.assignments.context.eq(context) & catalog.assignments.profile_id.eq(reference)]
    if not labels.empty:
        st.caption("Reference subgroup: " + str(labels.description.iloc[0]) + " · status: " + str(labels.status.iloc[0]))
    if peers.empty:
        st.info("No supported specification peers are available for this reference. Sparse or unresolved profiles remain unassigned.")
        return
    columns = ["rank", *COHORT, "category", "powertrain", "specification_distance", "description", "status", "shared_cluster", "vin_count"]
    st.dataframe(peers[[c for c in columns if c in peers]], hide_index=True, width="stretch",
                 alt="V6 direct specification peers with cluster evidence from the reference's held-out fit")
    st.caption("One result per Make/Model family. All subgroup labels use the reference's fitted context: " + context + ". Direct fallback means no accepted cluster assignment; it does not prevent specification matching.")
    st.download_button("Download similar vehicles", peers.to_csv(index=False), "similar_vehicles_v6.csv", key=key + "_csv")
    with st.expander("Compare specifications"):
        selected = choose_profiles(pd.concat([reference_row[catalog.profiles.columns], peers[catalog.profiles.columns]], ignore_index=True), key + "_compare")
        if selected:
            path = catalog.specs / grain / "contexts" / context
            model = joblib.load(path / "retrieval.joblib")
            fields = [field for field in model["fields"] if field in NUMERIC]
            comparison_chart(catalog.profiles.set_index("profile_id").reindex(selected), fields,
                             profile_names(catalog.profiles), key + "_specifications", SPEC_LABELS)


def profile_names(profiles, identity="profile_id"):
    columns = [c for c in COHORT if c in profiles]
    result = profiles.copy()
    result["display_name"] = result[columns].astype("string").fillna("").agg(" ".join, axis=1).str.strip()
    if identity == "vehicle_id" and "variant_description" in result:
        result["display_name"] = result.variant_description.fillna(result.display_name)
    if identity == "family_id":
        return dict(zip(result[identity], result.display_name))
    if "category" in result:
        result["display_name"] += " · " + result.category.astype(str)
    result["display_name"] += " · " + result[identity].astype(str).str[:8]
    return dict(zip(result[identity], result.display_name))


def choose_profiles(profiles, key, maximum=3, identity="profile_id"):
    names = profile_names(profiles, identity)
    previous = [v for v in st.session_state.get(key, []) if v in names]
    if key in st.session_state and previous != st.session_state[key]:
        st.session_state[key] = previous
    return st.multiselect("Compare up to three vehicles", list(names), default=None,
                          format_func=names.get, max_selections=maximum, key=key)


def discovery_display(key, safety=False):
    previous = st.session_state.get(key)
    if previous in ("Map", "Compare"):
        st.session_state[key] = {"Map": "Overview", "Compare": "Compare vehicles"}[previous]
    options = ["Overview", "Compare vehicles"] + ([] if safety else ["Similar vehicles"]) + ["Advanced exploration"]
    return st.segmented_control("Display", options, default="Overview", key=key) or "Overview"


def map_chart(points, arrows, selected, key, label="Vehicle locations"):
    if points.empty or not {"x", "y", "entity_id"}.issubset(points):
        st.info("Not enough supported coordinates for a two-dimensional map.")
        return
    points = points.dropna(subset=["x", "y"]).copy()
    total = len(points)
    if total > 3000:
        pinned = points.loc[points.entity_id.isin(selected)]
        rest = points.loc[~points.entity_id.isin(selected)].sort_values("entity_id").sample(3000 - len(pinned), random_state=42)
        points = pd.concat([pinned, rest], ignore_index=True)
    points["role"] = np.where(points.entity_id.isin(selected), "Selected", "Other")
    color = "cluster_label" if "cluster_label" in points else "role"
    tooltip = [c for c in [*COHORT, "vehicle_id", "category", "cluster_label", "matching_listings", "entity_id", "usable_documents", "supported", "x", "y"] if c in points]
    chart = alt.Chart(points).mark_point(filled=True, size=65).encode(
        x=alt.X("x:Q", title="Coordinate 1"), y=alt.Y("y:Q", title="Coordinate 2"),
        color=alt.Color(color + ":N"), tooltip=tooltip,
        shape=alt.Shape("role:N", scale=alt.Scale(domain=["Other", "Selected"], range=["circle", "diamond"])))
    if not arrows.empty:
        drawn = arrows.copy()
        extent = max(float(points.x.abs().quantile(.95)), float(points.y.abs().quantile(.95)), 1e-8)
        norm = max(float(np.linalg.norm(drawn[["x", "y"]], axis=1).max()), 1e-8)
        scale = .8 * extent / norm
        drawn["end_x"], drawn["end_y"] = drawn.x * scale, drawn.y * scale
        drawn["zero"] = 0.
        drawn["angle"] = 90 - np.degrees(np.arctan2(drawn.y, drawn.x))
        tips = ["label:N", alt.Tooltip("x:Q", title="Arrow coordinate 1", format=".4f"),
                alt.Tooltip("y:Q", title="Arrow coordinate 2", format=".4f"), "definition:N"]
        rules = alt.Chart(drawn).mark_rule(color="#ad6d20").encode(x="zero:Q", y="zero:Q", x2="end_x:Q", y2="end_y:Q", tooltip=tips)
        heads = alt.Chart(drawn).mark_point(shape="triangle", filled=True, color="#ad6d20", size=70).encode(
            x="end_x:Q", y="end_y:Q", angle=alt.Angle("angle:Q", scale=None), tooltip=tips)
        labels = alt.Chart(drawn).mark_text(dx=6, dy=-8, align="left", color="#ad6d20").encode(x="end_x:Q", y="end_y:Q", text="label:N", tooltip=tips)
        chart = chart + rules + heads + labels
        st.caption("Arrow lengths share one display scale; hover shows their original values. Directions describe features or themes, not safety or reliability.")
    st.altair_chart(chart.properties(height=420).interactive(), key=key, alt=label)
    st.caption(f"{len(points):,} of {total:,} locations shown. Filtering does not refit the projection.")


def measure_cells(frame, fields, names, labels=None, bounds=None, percentages=False):
    """Keep observed zero, missing evidence and display-only normalization distinct."""
    labels = labels or {}
    rows = []
    for identity, row in frame.iterrows():
        for position, field in enumerate(fields):
            value = pd.to_numeric(row.get(field), errors="coerce")
            value = float(value) if pd.notna(value) and np.isfinite(value) else np.nan
            shade = value
            if bounds is not None and np.isfinite(value):
                low, high = bounds[0][position], bounds[1][position]
                if low is None or high is None or not np.isfinite([low, high]).all():
                    shade = np.nan
                else:
                    shade = float(np.clip((value - low) / (high - low), 0, 1)) if high > low else .5
            rows.append({"entity": str(names.get(identity, identity)), "measure": labels.get(field, field.replace("_", " ")),
                         "value": value, "shade": shade,
                         "display": "Unknown" if not np.isfinite(value) else (f"{value:.1%}" if percentages else f"{value:,.1f}".removesuffix(".0")),
                         **{field: row[field] for field in ("record_count", "usable_documents", "supported", "profile_count", "observed_profiles") if field in row}})
            if field + "_observed" in row:
                rows[-1]["observed_profiles"] = row[field + "_observed"]
    return pd.DataFrame(rows)


def heatmap_chart(frame, fields, names, key, labels=None, bounds=None, percentages=False, domain=None, color_title="Value"):
    fields = [field for field in fields if field in frame]
    if frame.empty or not fields:
        st.info("No saved measures are available for this selection.")
        return
    cells = measure_cells(frame, fields, names, labels, bounds, percentages)
    domain = [0, 1] if bounds is not None or percentages else (domain or [0, max(1, cells.value.max())])
    tooltip = [alt.Tooltip("entity:N", title="Vehicle / subgroup"), alt.Tooltip("measure:N", title="Measure"),
               alt.Tooltip("display:N", title="Recorded value")]
    tooltip += [alt.Tooltip(field + ":Q", title=field.replace("_", " ").capitalize())
                for field in ("record_count", "usable_documents", "profile_count", "observed_profiles") if field in cells]
    if "supported" in cells:
        tooltip.append(alt.Tooltip("supported:N", title="Text support threshold met"))
    base = alt.Chart(cells).encode(
        x=alt.X("measure:N", sort=list(dict.fromkeys(cells.measure)), title=None,
                axis=alt.Axis(orient="top", labelAngle=-35, labelLimit=180, labelOverlap=False)),
        y=alt.Y("entity:N", sort=list(dict.fromkeys(cells.entity)), title=None, axis=alt.Axis(labelLimit=250, labelOverlap=False)), tooltip=tooltip)
    tiles = base.mark_rect().encode(color=alt.condition("isValid(datum.shade)",
        alt.Color("shade:Q", scale=alt.Scale(scheme="blues", domain=domain, clamp=True),
                  legend=alt.Legend(title=color_title, format=".0%" if percentages else (".1~f" if bounds is not None else ",.0f"))), alt.value("#e5e7eb")))
    threshold = domain[0] + .55 * (domain[1] - domain[0])
    text = base.mark_text(fontSize=11).encode(text="display:N", color=alt.condition(
        f"isValid(datum.shade) && datum.shade > {threshold}", alt.value("white"), alt.value("#172b4d")))
    st.altair_chart((tiles + text).properties(height=150 + 32 * len(frame)), key=key,
                   alt="Labeled saved measures by vehicle or specification subgroup; missing cells say Unknown")


def comparison_chart(frame, fields, names, key, labels=None, percentages=False, domain=None):
    fields = [field for field in fields if field in frame]
    if frame.empty or not fields:
        st.info("Select up to three vehicles with saved measures for comparison.")
        return
    cells = measure_cells(frame, fields, names, labels, percentages=percentages)
    for offset in range(0, len(fields), 2):
        for column, field in zip(st.columns(2), fields[offset:offset + 2]):
            with column:
                label = (labels or {}).get(field, field.replace("_", " "))
                data = cells.loc[cells.measure.eq(label)]
                base = alt.Chart(data).encode(y=alt.Y("entity:N", sort=data.entity.tolist(), title=None,
                    axis=alt.Axis(labelLimit=240)), tooltip=[alt.Tooltip("entity:N", title="Vehicle"),
                    alt.Tooltip("display:N", title=label)])
                scale = alt.Scale(domain=domain, zero=True) if domain is not None else alt.Scale(zero=True)
                points = base.transform_filter("isValid(datum.value)").encode(
                    x=alt.X("value:Q", title=None, scale=scale, axis=alt.Axis(format=".0%" if percentages else ",.0f")))
                chart = points.mark_point(filled=True, size=90, color="#0072b2") + points.mark_text(align="left", dx=9).encode(text="display:N")
                chart += base.transform_filter("!isValid(datum.value)").mark_text(align="left").encode(x=alt.value(6), text="display:N")
                st.altair_chart(chart.properties(title=label, height=65 + 36 * len(frame), padding={"right": 35}), key=key + "_" + field,
                               alt=label + " for selected vehicles; missing observations are labeled Unknown")


def overview_page(frame, key, selected=(), size=15):
    """Bound chart density without silently sampling or dropping selected rows."""
    pinned = frame.loc[frame.index.isin(selected)]
    rest = frame.loc[~frame.index.isin(selected)]
    pages = max(1, int(np.ceil(len(rest) / size)))
    scope = (tuple(frame.index), tuple(selected))
    if st.session_state.get(key + "_scope") != scope:
        st.session_state[key + "_scope"] = scope
        st.session_state.pop(key + "_page", None)
    page = int(st.number_input("Overview page", min_value=1, max_value=pages, value=1, step=1, key=key + "_page")) if pages > 1 else 1
    result = pd.concat([pinned, rest.iloc[(page - 1) * size:page * size]])
    st.caption(f"Showing {len(result):,} of {len(frame):,} matching rows · page {page} of {pages}. Selected vehicles stay visible.")
    return result


def ranked_chart(frame, identity, score, names, key, title):
    if frame.empty:
        st.info("No supported peers are available for this reference within the current filters.")
        return
    frame = frame.copy()
    frame["vehicle"] = frame[identity].map(names)
    base = alt.Chart(frame).encode(y=alt.Y("vehicle:N", sort=frame.vehicle.tolist(), title=None, axis=alt.Axis(labelLimit=360)),
        x=alt.X(score + ":Q", title=title, scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("vehicle:N", title="Vehicle"), alt.Tooltip(score + ":Q", title=title, format=".3f")])
    chart = base.mark_bar(color="#0072b2") + base.mark_text(align="left", dx=5).encode(text=alt.Text(score + ":Q", format=".3f"))
    st.altair_chart(chart.properties(height=max(90, len(frame) * 32)), key=key, alt=title + " for supported peers of the selected reference")


def text_neighbors(path, profiles, reference, limit=10):
    """Rank supported families in one saved reference-fold vector space."""
    features = pd.read_parquet(path / "family_features.parquet")
    index = pd.read_parquet(path / "family_vector_index.parquet")
    rows = profiles.merge(features, on="family_id", validate="1:1").merge(index, on="family_id", validate="1:1")
    query = rows.loc[rows.family_id.eq(reference) & rows.supported]
    if query.empty:
        return pd.DataFrame()
    candidates = rows.loc[rows.supported & rows.identity_complete & rows.family_block.ne(query.family_block.iloc[0])].copy()
    vectors = np.load(path / "family_vectors.npy", mmap_mode="r", allow_pickle=False)
    reference_vector = np.asarray(vectors[int(query.vector_row.iloc[0])], dtype=float)
    values = np.asarray(vectors[candidates.vector_row.to_numpy(int)], dtype=float)
    norms = np.linalg.norm(values, axis=1) * np.linalg.norm(reference_vector)
    candidates["cosine_similarity"] = np.divide(values @ reference_vector, norms,
        out=np.full(len(values), np.nan), where=norms > 1e-12).clip(-1, 1)
    return candidates.dropna(subset=["cosine_similarity"]).sort_values(
        ["cosine_similarity", "family_id"], ascending=[False, True]).head(limit)


def projection_caption(path):
    metadata = a.read_json(path / "projection.json")
    ratio = metadata.get("explained_variance_ratio")
    if ratio:
        st.caption("Displayed variance: " + ", ".join(f"axis {i + 1} {v:.1%}" for i, v in enumerate(ratio)) + ". Two dimensions do not preserve every original distance.")
    return metadata


def specification_summary(profiles, fields):
    profiles = profiles.copy()
    profiles["subgroup"] = profiles.apply(lambda row:
        f"Group {int(row.cluster) + 1}: {row.description}" if pd.notna(row.cluster) and row.cluster >= 0
        else ("Partial profiles" if row.status == "insufficient_observations" or pd.isna(row.status)
              else "Direct peers (no accepted subgroup)"), axis=1)
    groups = profiles.groupby("subgroup", sort=True)
    summary = groups[fields].median()
    summary["profile_count"] = groups.size()
    for field in fields:
        summary[field + "_observed"] = groups[field].count()
    return profiles, summary


def category_dashboard(source, filters, key="v6_categories"):
    catalog = active_catalog(source)
    if catalog is None:
        return False
    grain = st.segmented_control("Discovery population", ["model", "cohort"], default="model", key=key + "_grain",
                                 format_func=lambda x: "Model level" if x == "model" else "Trim cohorts")
    catalog = active_catalog(source, grain or "model")
    st.subheader("Vehicle categories")
    discovery_caption(catalog)
    profiles = catalog.scope(source, filters)
    if profiles.empty:
        st.info("No vehicle profiles match the dashboard filters.")
        return True
    st.metric("Matching listings", f"{int(profiles.matching_listings.sum()):,}")
    categories = sorted(profiles.category.unique())
    common = profiles.loc[profiles.category.ne("Body unresolved")].groupby("category").matching_listings.sum()
    default_category = common.idxmax() if not common.empty else categories[0]
    category = st.selectbox("Body category", categories, index=categories.index(default_category), key=key + "_body")
    profiles = profiles.loc[profiles.category.eq(category)]
    selected = choose_profiles(profiles, key + "_selected")
    reference = selected[0] if selected else profiles.profile_id.iloc[0]
    context = catalog.context(profiles.set_index("profile_id").loc[reference])
    rows = catalog.assignments.loc[catalog.assignments.context.eq(context)]
    profiles = profiles.merge(rows[["profile_id", "eligible", "cluster", "description", "status"]], on="profile_id", how="left", validate="1:1")
    with st.container(horizontal=True):
        st.metric("Supported profiles", int(profiles.eligible.fillna(False).sum()))
        st.metric("Partial profiles", int((~profiles.eligible.fillna(False)).sum()))
    path = catalog.specs / catalog.grain / "contexts" / context / "projection"
    model_path = path.parent / "retrieval.joblib"
    model = joblib.load(model_path) if model_path.exists() else {"fields": []}
    fields = [field for field in model["fields"] if field in NUMERIC]
    metadata = a.read_json(path / "projection.json") if (path / "projection.json").exists() else {}
    bounds = (metadata["radar_min"], metadata["radar_max"]) if "radar_min" in metadata else None
    profiles, summary = specification_summary(profiles, fields)
    mode = discovery_display(key + "_display")
    if mode == "Overview":
        st.caption("Subgroup medians weight matching profiles equally; n counts profiles, not listings. Group IDs belong to the current saved reference fit and can change with the selected vehicle. Partial profiles and direct-peer fallbacks stay separate.")
        names = {identity: f"{identity.split(':')[0].replace(' (no accepted subgroup)', '')} · n={int(row.profile_count):,}"
                 for identity, row in summary.iterrows()}
        if not fields:
            st.info("No numeric specification fields met the saved support checks for this body category. Profile counts and source details remain available.")
        elif bounds is not None:
            heatmap_chart(summary, fields, names, key + "_overview", SPEC_LABELS, bounds=bounds,
                          color_title="Reference range")
            st.caption("Color uses frozen development 1st–99th percentile ranges within each specification. Cell values retain original units; larger does not mean better. Unknown cells have no supported observation.")
        else:
            comparison_chart(summary, fields, names, key + "_overview", SPEC_LABELS)
    elif mode == "Compare vehicles":
        comparison_chart(profiles.set_index("profile_id").reindex(selected), fields,
                         profile_names(catalog.profiles), key + "_compare", SPEC_LABELS)
        st.caption("Each specification has its own axis in original units. Missing observations stay unknown.")
    elif mode == "Similar vehicles":
        if not selected:
            st.info("Select a reference vehicle above to find specification peers.")
        else:
            peers, _ = catalog.similar(selected[0], limit=10, candidate_ids=profiles.profile_id)
            ranked_chart(peers, "profile_id", "specification_distance", profile_names(catalog.profiles),
                         key + "_peers", "Specification distance (smaller is closer)")
            st.caption("Reference: " + profile_names(catalog.profiles)[selected[0]] + ". Other Make/Model families within current filters, the same body category and ±10 model years; one reference-held-out fit.")
    else:
        if (path / "points.parquet").exists():
            projection_caption(path)
            points = pd.read_parquet(path / "points.parquet").merge(profiles, left_on="entity_id", right_on="profile_id", validate="1:1")
            points["cluster_label"] = points.subgroup
            show_arrows = st.checkbox("Show feature directions", value=False, key=key + "_arrows")
            arrows = pd.read_parquet(path / "arrows.parquet") if show_arrows else pd.DataFrame()
            map_chart(points, arrows, selected, key + "_map")
        else:
            st.info("This body category has insufficient supported specification data for a saved projection.")
    if selected:
        row = catalog.profiles.set_index("profile_id").loc[selected[0]]
        st.markdown("**" + profile_names(catalog.profiles)[selected[0]] + "**")
        details = [f"{row[f]:.0f} {unit}" for f, unit in (("seating", "seats"), ("horsepower", "hp"), ("length", "in long"), ("wheelbase", "in wheelbase"))
                   if pd.notna(row.get(f))]
        st.caption(" · ".join(details))
        st.caption("Profile scope: " + str(row.fallback_level).replace("_", " ") + " · propulsion evidence: " + str(row.powertrain).replace("_", " "))
    with st.expander("Research details"):
        if st.checkbox("Show profile details", key=key + "_details"):
            st.dataframe(profiles[[c for c in [*COHORT, "description", "status", "matching_listings", "profile_id"] if c in profiles]], hide_index=True,
                         width="stretch", alt="Selected body-category profiles and evidence states")
        st.download_button("Download matching profiles", profiles.to_csv(index=False), "vehicle_profiles.csv", key=key + "_csv")
    return True


@st.cache_data(max_entries=32, show_spinner=False)
def descriptive_themes(directory, receipt_key):
    """Describe saved issue groups with distinctive terms from their source text."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    path = Path(directory)
    members = pd.read_parquet(path / "memberships.parquet")
    members = members.loc[members.usable & members.theme.ge(0)].drop_duplicates("document_id")
    # Bound display-only term extraction; never refit the saved issue groups.
    members = members.sort_values("document_id").sample(frac=1, random_state=42).groupby("theme").head(200)
    field = {"complaint": "summary", "defect": "summary", "hazard": "consequence", "remedy": "remedy"}[path.parent.parent.name]
    documents = (pl.scan_parquet(path.parent.parent / "documents.parquet")
                 .filter(pl.col("document_id").is_in(members.document_id.tolist()))
                 .select("document_id", field).collect().to_pandas())
    rows = members.merge(documents, on="document_id", validate="1:1")
    if rows.empty:
        return {}
    texts = rows.groupby("theme")[field].agg(lambda values: " ".join(values.fillna("").astype(str)))
    try:
        encoder = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", max_features=10000, sublinear_tf=True)
        weights = encoder.fit_transform(texts).toarray()
    except ValueError:
        return {}
    terms = encoder.get_feature_names_out()
    labels = {}
    for theme, weights_row in zip(texts.index, weights):
        top = [terms[i] for i in np.argsort(-weights_row)[:4] if weights_row[i] > 0]
        labels[int(theme)] = f"Theme {theme}: " + ", ".join(top)
    return labels


def descriptive_theme_arrows(directory, receipt_key):
    arrows = pd.read_parquet(Path(directory) / "arrows.parquet").copy()
    if not arrows.label.str.startswith("Theme ").any():
        return arrows
    for theme, label in descriptive_themes(directory, receipt_key).items():
        mask = arrows.label.eq(f"Theme {theme}")
        arrows.loc[mask, "label"] = label
        arrows.loc[mask, "definition"] = "Projected issue-group prototype; distinctive terms from up to 200 saved narratives: " + label
    return arrows


def nhtsa_dashboard(source, filters, key="v6_nhtsa", view=None):
    catalog = active_catalog(source)
    if catalog is None:
        return False
    st.subheader("NHTSA evidence")
    discovery_caption(catalog)
    if view is None:
        view = st.segmented_control("Evidence", ["Complaints", "Recalls", "Safety"], default="Complaints", key=key + "_source") or "Complaints"
    profiles = catalog.scope(source, filters)
    families = set(profiles.family_id)
    if view == "Safety":
        safety = catalog.safety.loc[catalog.safety.family_id.isin(families)].copy()
        if safety.empty:
            st.info("No saved safety variants match this selection.")
            return True
        era = st.selectbox("Rating system", sorted(safety.rating_era.unique()), key=key + "_era")
        safety = safety.loc[safety.rating_era.eq(era)]
        selected = choose_profiles(safety, key + "_safety_selected", identity="vehicle_id")
        mode = discovery_display(key + "_safety_display", safety=True)
        path = catalog.path / "safety_projections" / era
        st.caption("Saved tested variants in one rating era. Side, pole and rollover ratings support cross-class display. Overall/frontal comparisons require the same class and supported ±250 lb weight context; see the source tables for those ratings.")
        names = profile_names(safety, "vehicle_id")
        safety = safety.sort_values(["Make", "Model", "ModelYear", "vehicle_id"]).set_index("vehicle_id", drop=False)
        if mode == "Overview":
            page = overview_page(safety, key + "_safety", selected)
            heatmap_chart(page, COMPARABLE_RATINGS, names, key + "_safety_overview", RATING_LABELS,
                          domain=[1, 5], color_title="Recorded stars")
            st.caption("Unknown means no recorded valid rating for this measure. This snapshot does not reconstruct historical rating versions.")
        elif mode == "Compare vehicles":
            comparison_chart(safety.reindex(selected), COMPARABLE_RATINGS, names, key + "_safety_compare",
                             RATING_LABELS, domain=[0, 5])
        elif path.exists():
            metadata = projection_caption(path)
            if metadata.get("status") == "completed":
                points = pd.read_parquet(path / "points.parquet").merge(safety.reset_index(drop=True), left_on="entity_id", right_on="vehicle_id", validate="1:1")
                show_arrows = st.checkbox("Show rating directions", value=False, key=key + "_safety_arrows")
                arrows = pd.read_parquet(path / "arrows.parquet") if show_arrows else pd.DataFrame()
                map_chart(points, arrows, selected, key + "_safety_map", "Structured crash-test rating locations")
            else:
                st.info("The saved safety projection has insufficient supported measures. Recorded star ratings remain available in Overview and Compare vehicles.")
        else:
            st.info("No saved safety projection is available for this rating era.")
        return True
    default_period = "latest_saved" if "latest_saved" in catalog.receipt["periods"] else catalog.receipt["periods"][0]
    period = st.segmented_control("Evidence period", catalog.receipt["periods"], default=default_period,
                                  format_func=lambda p: "Historical cutoff" if p == "historical" else "Latest saved", key=key + "_period") or default_period
    representation = st.segmented_control("Text representation", ["tfidf", "embeddings"], default="tfidf", key=key + "_representation") or "tfidf"
    channel = "complaint" if view == "Complaints" else st.selectbox("Recall text", ["defect", "hazard", "remedy"], key=key + "_channel")
    family_profiles = profiles.drop_duplicates("family_id").sort_values(["Make", "Model", "ModelYear"], na_position="last")
    selected_families = choose_profiles(family_profiles.drop(columns=["Trim", "category"], errors="ignore"), key + "_selected", identity="family_id")
    reference = family_profiles.set_index("family_id", drop=False).loc[selected_families[0]] if selected_families else (family_profiles.iloc[0] if not family_profiles.empty else None)
    if reference is None:
        st.info("No vehicle families match the filters.")
        return True
    path = catalog.text / period / channel / ("fold_" + str(int(reference.fold))) / representation
    audit = a.read_json(catalog.text / "manifest.json")["audits"][period]
    st.caption(f"Evidence cutoff: {audit['cutoff']} · latest saved record: {audit['maximum_evidence_date']} · collection freshness unverified. Reports and campaigns do not establish failure rates.")
    names = profile_names(family_profiles, "family_id")
    mode = discovery_display(key + "_display")
    text_available = (path / "family_features.parquet").exists()
    if mode in ("Overview", "Compare vehicles"):
        pattern = st.segmented_control("Pattern", ["Reported components", "Narrative themes"], default="Reported components", key=key + "_pattern") or "Reported components"
        component = catalog.components(period)
        component = component.loc[component.source.eq("complaint" if channel == "complaint" else "recall")]
        fields = [c + ("_share" if channel == "complaint" else "_count") for c in COMPONENTS]
        labels = {field: name.capitalize() for field, name in zip(fields, COMPONENTS)}
        percentages = channel == "complaint"
        values = component
        if pattern == "Narrative themes":
            if text_available:
                values = pd.read_parquet(path / "family_features.parquet")
                fields = sorted(field for field in values if field.startswith("theme_") and field.endswith("_share"))
                descriptions = descriptive_themes(str(path), stat_key(path / "memberships.parquet"))
                labels = {field: descriptions.get(int(field.split("_")[1]), "Theme " + field.split("_")[1]) for field in fields}
                fields = st.multiselect("Themes to display", fields, default=fields[:8], format_func=labels.get,
                                        key=key + "_themes_" + a.digest(str(path))[:12])
                if not fields:
                    st.info("Select at least one saved narrative theme to display.")
                percentages = True
                st.caption("Shares of usable saved narratives assigned to each issue group. Unassigned narratives remain in the denominator. Theme descriptions are automatic and unreviewed; themes are local to this text context.")
            else:
                values, fields = pd.DataFrame(columns=["family_id"]), []
                st.info("Insufficient usable text or training vocabulary for saved narrative themes. Reported components remain available.")
        else:
            st.caption("Complaint cells show shares of saved reports; recall cells show campaign counts. Components can overlap. Unknown means no saved source evidence; a recorded zero is distinct from unknown.")
        frame = family_profiles.merge(values, on="family_id", how="left", validate="1:1").set_index("family_id")
        frame = frame.reindex(columns=list(dict.fromkeys([*frame.columns, *fields])))
        if fields:
            if mode == "Overview":
                page = overview_page(frame, key + "_" + pattern, selected_families)
                heatmap_chart(page, fields, names, key + "_overview", labels, percentages=percentages,
                              domain=[0, max(1, frame[fields].max().max())],
                              color_title="Narrative share" if pattern == "Narrative themes" else ("Report share" if percentages else "Campaigns"))
            else:
                comparison_chart(frame.reindex(selected_families), fields, names, key + "_compare", labels,
                                 percentages=percentages, domain=[0, 1] if percentages else None)
            support = page if mode == "Overview" else frame.reindex(selected_families)
            support_fields = [field for field in ("record_count", "usable_text_count", "usable_documents", "supported") if field in support]
            if not support.empty:
                details = support[[*a.FAMILY, *support_fields]].rename(columns={"ModelYear": "Model year",
                    "record_count": "Saved reports" if channel == "complaint" else "Saved campaigns",
                    "usable_text_count": "Usable summaries", "usable_documents": "Usable narratives",
                    "supported": "Text support threshold met"})
                st.dataframe(details, hide_index=True, width="stretch",
                             alt="Evidence denominators and text support for displayed vehicle families")
    elif mode == "Similar vehicles":
        if not selected_families:
            st.info("Select a reference vehicle above to find similar saved narratives.")
        elif text_available:
            peers = text_neighbors(path, family_profiles, selected_families[0])
            ranked_chart(peers, "family_id", "cosine_similarity", names, key + "_text_peers", "Narrative cosine similarity (larger is closer)")
            st.caption("Reference: " + names[selected_families[0]] + ". Other Make/Model families within current filters, using the same saved reference-fold vectors. At least 20 usable complaints or one usable recall summary are required. Similarity is not a probability, safety score or failure rate.")
        else:
            st.info("No supported saved text vectors are available for this selection.")
    elif (path / "family_points.parquet").exists():
        projection_caption(path)
        points = pd.read_parquet(path / "family_points.parquet").merge(family_profiles, left_on="entity_id", right_on="family_id", validate="1:1")
        show_arrows = st.checkbox("Show word / theme directions", value=False, key=key + "_arrows")
        arrows = descriptive_theme_arrows(str(path), stat_key(path / "arrows.parquet")) if show_arrows else pd.DataFrame()
        st.caption("Each point is a model-year/make/model family, pooling trims. Diamonds mark selected families. The two-dimensional projection does not preserve every original distance.")
        map_chart(points, arrows, selected_families, key + "_text_map", "Family text locations and optional feature directions")
        if show_arrows:
            st.caption("TF-IDF arrows show term loadings; embedding arrows show saved issue-group prototypes. Axes and theme IDs differ across periods, channels, representations and reference folds.")
            st.dataframe(arrows[["label", "definition"]].rename(columns={"label": "Map label", "definition": "Meaning"}), hide_index=True,
                         width="stretch", alt="Descriptions of terms and issue themes shown on the projection")
    else:
        st.info("No saved text projection is available for this selection.")
    if not selected_families:
        st.caption("Select a vehicle above to inspect its representative saved evidence.")
        return True
    record_path = catalog.text / period / "records.parquet"
    records = family_records(str(catalog.text), period, (reference.family_id,), stat_key(record_path))
    records = records.loc[records.family_id.eq(reference.family_id) & records.source.eq("complaint" if channel == "complaint" else "recall")]
    field = {"complaint": "summary", "defect": "summary", "hazard": "consequence", "remedy": "remedy"}[channel]
    if not records.empty and text_available:
        # Central examples are selected in the saved text geometry, once per source record.
        mapping = pd.read_parquet(path / "memberships.parquet")
        support = mapping.loc[mapping.family_id.eq(reference.family_id) & mapping.usable]
        vectors = np.load(a.vector_path(path), mmap_mode="r", allow_pickle=False)
        indices = support.document_row.to_numpy(int)
        example_ids = []
        if len(indices):
            centroid = np.asarray(vectors[indices]).mean(axis=0)
            similarity = np.asarray(vectors[indices]) @ centroid
            example_ids = support.iloc[np.argsort(-similarity)[:3]].document_id.tolist()
        examples = records.loc[records.document_id.isin(example_ids)] if example_ids else records.head(3)
        st.markdown("**Representative saved evidence for the selected vehicle family**")
        for _, record in examples.drop_duplicates("record_id").iterrows():
            st.markdown("**" + str(record.record_id) + "** · " + str(record.evidence_date)[:10])
            st.write(record[field])
            st.link_button("Open NHTSA source", "https://www.nhtsa.gov/recalls?nhtsaId=" + str(record.source_id))
    with st.expander("Research details"):
        if st.checkbox("Show saved source rows", key=key + "_details"):
            st.dataframe(records.head(500), hide_index=True, width="stretch", alt="Saved applicable NHTSA records for the selected family")
        st.download_button("Download selected family evidence", records.to_csv(index=False), "nhtsa_family_evidence.csv", key=key + "_csv")
    return True


def diverse_top(frame, order, ascending, limit=5):
    if frame.empty:
        return frame
    frame = frame.sort_values(order + ["price", "vin"], ascending=ascending + [True, True], na_position="last")
    first = frame.drop_duplicates("family_id").head(limit)
    if len(first) < limit:
        first = pd.concat([first, frame.loc[~frame.vin.isin(first.vin)].drop_duplicates(a.FAMILY + ["Trim"]).head(limit - len(first))])
    return first


def text_context(catalog, family_id, fold, period, reference_family=None):
    """Expose real text coordinates/mixtures and original-space similarity, never a risk score."""
    from sklearn.preprocessing import normalize

    result = []
    for channel in ("complaint", "defect", "hazard", "remedy"):
        for representation in ("tfidf", "embeddings"):
            path = catalog.text / period / channel / f"fold_{fold}" / representation
            if not (path / "family_features.parquet").exists():
                continue
            feature = pd.read_parquet(path / "family_features.parquet")
            feature = feature.loc[feature.family_id.eq(family_id)]
            if feature.empty:
                continue
            item = feature.iloc[0].to_dict()
            item.update(channel=channel, representation=representation, context=str(path.relative_to(catalog.text)),
                        source_scope="model-year family", coordinates_display_only=True)
            points = pd.read_parquet(path / "family_points.parquet")
            point = points.loc[points.entity_id.eq(family_id)]
            if not point.empty:
                item.update(x=float(point.x.iloc[0]), y=float(point.y.iloc[0]))
            memberships = pd.read_parquet(path / "memberships.parquet")
            wanted = {family_id} | ({reference_family} if reference_family else set())
            members = memberships.loc[memberships.family_id.isin(wanted) & memberships.usable].drop_duplicates(["family_id", "record_id"])
            vectors = np.load(a.vector_path(path), mmap_mode="r", allow_pickle=False)
            centroids = {}
            for family, group in members.groupby("family_id"):
                indices = group.document_row.to_numpy(int)
                if len(indices):
                    centroid = np.asarray(vectors[indices]).mean(axis=0)
                    centroids[family] = normalize(centroid.reshape(1, -1))[0]
                    if family == family_id:
                        order = np.argsort(-(np.asarray(vectors[indices]) @ centroids[family]))[:3]
                        example_ids = group.iloc[order].document_id.tolist()
                        example = pl.scan_parquet(path.parent.parent / "documents.parquet").filter(pl.col("document_id").is_in(example_ids)).collect().to_pandas()
                        item["representative_records"] = [{"record_id": str(r.record_id), "evidence_date": str(r.evidence_date),
                                                           "text": str(r[{"complaint": "summary", "defect": "summary", "hazard": "consequence", "remedy": "remedy"}[channel]]),
                                                           "source_id": str(r.source_id)} for _, r in example.iterrows()]
            if reference_family in centroids and family_id in centroids:
                item["cosine_distance_to_reference"] = float(np.clip(1 - centroids[family_id] @ centroids[reference_family], 0, 2))
            result.append(item)
    return json.loads(json.dumps(result, default=str))


def applicable_safety(candidates, catalog, reference):
    if reference is None or reference.empty:
        return pd.DataFrame(), "Select a supported reference with curb weight to compare overall crash ratings."
    reference = reference.iloc[0]
    weight = reference.get("curb_weight_lb")
    if not pd.notna(weight) or reference.get("curb_weight_lb_n", 0) < 5 or reference.get("curb_weight_lb_n", 0) / max(reference.get("vin_count", 0), 1) < .9:
        return pd.DataFrame(), "The reference lacks supported curb weight; overall safety ranking is unavailable."
    def weight_class(row):
        body = row.get("body_family", "")
        if body != "passenger_car":
            return str(body)
        w = row.get("curb_weight_lb")
        if pd.isna(w) or w < 1500:
            return "unknown"
        return "car_" + str(min(int((w - 1500) // 500), 4))
    reference_class = weight_class(reference)
    if weight_class({**reference.to_dict(), "curb_weight_lb": reference.get("curb_weight_lb_min", weight)}) != weight_class({**reference.to_dict(), "curb_weight_lb": reference.get("curb_weight_lb_max", weight)}):
        return pd.DataFrame(), "The reference crosses a curb-weight class boundary; overall rating comparison is unavailable."
    rows = []
    for _, row in candidates.drop_duplicates("profile_id").iterrows():
        w = row.get("curb_weight_lb")
        if pd.isna(w) or row.get("curb_weight_lb_n", 0) < 5 or reference_class in ("unknown", "unknown_or_other", ""):
            continue
        coverage = row.get("curb_weight_lb_n", 0) / max(row.get("vin_count", 0), 1)
        if coverage < .9 or weight_class(row) != reference_class or abs(w - weight) > 250:
            continue
        if (int(row.ModelYear) >= 2011) != (int(reference.ModelYear) >= 2011):
            continue
        if weight_class({**row.to_dict(), "curb_weight_lb": row.get("curb_weight_lb_min", w)}) != weight_class({**row.to_dict(), "curb_weight_lb": row.get("curb_weight_lb_max", w)}):
            continue
        variants = catalog.safety.loc[catalog.safety.family_id.eq(row.family_id)]
        explicit = variants.loc[variants.variant_body.eq(row.category)]
        if explicit.empty:
            continue
        drive = row.get("wheel_system")
        explicit = explicit.loc[explicit.variant_drive.isna() | explicit.variant_drive.eq(drive)]
        if explicit.empty or explicit.OverallRating.isna().any() or explicit.OverallRating.nunique() != 1:
            continue
        rows.append({"profile_id": row.profile_id, "overall_rating": float(explicit.OverallRating.iloc[0]),
                     "safety_variants": ";".join(explicit.vehicle_id.astype(str)), "safety_scope": "compatible tested family variants; VIN equipment unverified"})
    if not rows:
        return pd.DataFrame(), "No candidates have sufficiently applicable and comparable overall crash-test evidence."
    return pd.DataFrame(rows), None


def search_lists(source, requirements, revision, config, legacy_catalog=None, *, include_text=True, limit=5):
    if not 1 <= limit <= 10:
        raise ValueError("Evidence shortlist limit must be between one and ten")
    from car_app.recommendation import search_candidates
    from car_app.model_registry import validate_price_run
    from car_app.artifacts import validate_sources

    catalog = active_catalog(source)
    if catalog is None:
        return None
    if requirements.get("schema_version") != 3:
        raise ValueError("Review the requirements again to use the new evidence-backed lists")
    period = requirements.get("evidence_period", "historical")
    if period not in catalog.receipt["periods"]:
        raise ValueError("Selected evidence period is unavailable")
    population = search_candidates(source, requirements, revision, catalog=legacy_catalog, return_population=True).select(
        "vin", "price", "mileage", *COHORT, "wheel_system").collect().to_pandas()
    population = population.drop(columns=["profile_id"], errors="ignore").merge(catalog.members, on="vin", how="left", validate="1:1")
    needed = ["profile_id", "family_id", "family_block", "category", "fold", "body_family", "curb_weight_lb", "curb_weight_lb_n",
              "curb_weight_lb_min", "curb_weight_lb_max", "vin_count", "fallback_level", "powertrain"]
    profile_columns = [c for c in needed if c in catalog.profiles and (c not in population or c == "profile_id")]
    population = population.merge(catalog.profiles[profile_columns], on="profile_id", how="left", validate="m:1")
    # The decoded listing columns remain the cohort identity; normalized keys belong to profiles.
    population["family_id"] = population.profile_id.map(catalog.profiles.set_index("profile_id").family_id)
    avoided = list(requirements.get("avoided_components", []))
    population["theme_group"] = 0
    if avoided:
        component = catalog.components(period)
        flags = component[["family_id", "supported", *[c + "_count" for c in avoided]]].copy()
        flags["match"] = flags[[c + "_count" for c in avoided]].sum(axis=1).gt(0)
        flags = flags.groupby("family_id").agg(supported=("supported", "all"), sources=("supported", "size"), match=("match", "any"))
        # Absence of a source block cannot establish theme absence.
        flags["group"] = np.where(flags.supported & flags.sources.ge(2), np.where(flags.match, 1, 0), 2)
        population["theme_group"] = population.family_id.map(flags.group).fillna(2).astype(int)
    unranked = population.loc[population.theme_group.eq(2)].copy()
    eligible = population.loc[population.theme_group.lt(2)].copy()
    order_prefix, asc_prefix = (["theme_group"], [True]) if avoided else ([], [])
    reference_id = requirements.get("reference_profile_id")
    reference = catalog.profiles.loc[catalog.profiles.profile_id.eq(reference_id)] if reference_id else None
    if reference_id and reference.empty:
        raise ValueError("The reference profile is no longer available; review the requirements again")
    spec = catalog.specification_peers(reference_id) if reference_id else pd.DataFrame(columns=["profile_id", "specification_distance"])
    spec = eligible.merge(spec, on="profile_id", validate="m:1")
    spec = diverse_top(spec, order_prefix + ["specification_distance"], asc_prefix + [True], limit)
    warnings, price = [], pd.DataFrame()
    active = config.get("active_price")
    if active:
        try:
            model_path = a.ROOT / "MODEL_OUTPUT" / active["run"]
            if a.file_hash(model_path / "run_metadata.json") != active["metadata_sha256"] or a.file_hash(model_path / active["model"] / "predictor.joblib") != active["predictor_sha256"]:
                raise ValueError("Pinned price artifacts changed")
            metadata = validate_price_run(model_path, active["model"])
            validate_sources(metadata)
            prediction = pd.read_parquet(model_path / (active["model"] + "_test_predictions.parquet"))
            split = pd.read_parquet(model_path / "split_manifest.parquet")
            test_vins = set(split.loc[split.partition.eq("test"), "vin"])
            if prediction.vin.duplicated().any() or not set(prediction.vin).issubset(test_vins):
                raise ValueError("Price ranking requires unique held-out test predictions")
            prediction = prediction[["vin", "prediction"]].rename(columns={"prediction": "predicted_price_usd"})
            price = eligible.merge(prediction, on="vin", validate="1:1")
            price = price.loc[np.isfinite(price.predicted_price_usd) & price.predicted_price_usd.gt(0)]
            price["model_relative_price"] = (price.predicted_price_usd - price.price) / price.predicted_price_usd
            price["price_model"] = active["run"] + "/" + active["model"]
            price = diverse_top(price, order_prefix + ["model_relative_price"], asc_prefix + [False], limit)
        except (OSError, ValueError, KeyError) as exc:
            warnings.append("Model-relative price list unavailable: " + str(exc))
    else:
        warnings.append("No compatible saved price model is selected.")
    safety, warning = applicable_safety(eligible, catalog, reference)
    if warning:
        warnings.append(warning)
    if not safety.empty:
        safety = diverse_top(eligible.merge(safety, on="profile_id", validate="m:1"), order_prefix + ["overall_rating"], asc_prefix + [False], limit)
    output = {"specifications": spec, "price": price, "safety": safety}
    # Source-backed issue mixtures and prototypes accompany each candidate for a future AI consumer.
    payload = {}
    reference_family = reference.family_id.iloc[0] if reference is not None and not reference.empty else None
    profile_index = catalog.profiles.set_index("profile_id")
    evidence_cache = {}
    for label, frame in output.items():
        if frame.empty:
            payload[label] = []
            continue
        components = catalog.components(period)
        for _, row in frame.iterrows():
            examples = components.loc[components.family_id.eq(row.family_id)]
            frame.loc[frame.vin.eq(row.vin), "issue_evidence"] = json.dumps(examples.replace({np.nan: None}).to_dict("records"), default=str)
            profile = profile_index.loc[row.profile_id]
            fold = int(reference.fold.iloc[0]) if reference is not None and not reference.empty else int(profile.fold)
            cache_key = (row.family_id, fold)
            if include_text and cache_key not in evidence_cache:
                evidence_cache[cache_key] = text_context(catalog, row.family_id, fold, period, reference_family)
            if include_text:
                frame.loc[frame.vin.eq(row.vin), "text_evidence"] = json.dumps(evidence_cache[cache_key], default=str)
            context = catalog.context(reference.iloc[0] if reference is not None and not reference.empty else profile)
            cluster = catalog.assignments.loc[catalog.assignments.context.eq(context) & catalog.assignments.profile_id.eq(row.profile_id)]
            if not cluster.empty:
                frame.loc[frame.vin.eq(row.vin), "specification_cluster"] = str(cluster.description.iloc[0])
                frame.loc[frame.vin.eq(row.vin), "cluster_status"] = str(cluster.status.iloc[0])
                frame.loc[frame.vin.eq(row.vin), "cluster_context"] = context
        columns = [c for c in [*COHORT, "vin", "profile_id", "family_id", "fallback_level", "price", "mileage", "specification_distance",
                   "model_relative_price", "predicted_price_usd", "price_model", "overall_rating", "safety_variants", "safety_scope", "theme_group", "issue_evidence",
                   "text_evidence", "specification_cluster", "cluster_status", "cluster_context"] if c in frame]
        payload[label] = json.loads(frame[columns].to_json(orient="records", date_format="iso"))
    return {"schema_version": 3, "ranked_lists": payload, "warnings": warnings, "eligible_listings": len(population),
            "unranked_insufficient_issue_evidence": int(len(unranked)), "evidence_period": period, "revision": revision,
            "reference_profile_id": reference_id, "avoided_components": avoided, "snapshot_identity": source.identity,
            "discovery_manifest": a.file_hash(catalog.path / "manifest.json"), "price_context": active,
            "api_calls_enabled": False}


def render_lists(result):
    st.subheader("Evidence-backed shortlists")
    st.caption(f"{result['eligible_listings']:,} eligible historical listings · evidence: {result['evidence_period']} · each list uses a separate ordering criterion.")
    label = st.segmented_control("Recommendation list", ["specifications", "price", "safety"], default="specifications",
                                 format_func=lambda x: {"specifications": "Closest specifications", "price": "Model-relative price", "safety": "Comparable crash tests"}[x], key="ranked_list_view")
    for warning in result["warnings"]:
        st.caption(warning)
    rows = result["ranked_lists"][label or "specifications"]
    if not rows:
        st.info("No supported results for this criterion. Choose a reference or inspect the evidence coverage.")
    for index, row in enumerate(rows, 1):
        with st.container(border=True):
            name = " ".join(str(row.get(c) or "") for c in COHORT)
            st.markdown(f"**{index}. {name}**")
            st.caption(f"Historical advertised price: ${row['price']:,.0f}" + (f" · mileage: {row['mileage']:,.0f} source units" if row.get("mileage") is not None else ""))
            if label == "specifications":
                st.metric("Specification distance", f"{row['specification_distance']:.3f}", help="Lower distances indicate closer specifications within the saved comparison context.")
            elif label == "price":
                with st.container(horizontal=True):
                    st.metric("Held-out predicted price", f"${row['predicted_price_usd']:,.0f}")
                    st.metric("Model-relative price signal", f"{row['model_relative_price']:.1%}")
                st.caption("A difference from the model estimate is not a verified discount or condition assessment.")
            else:
                st.metric("Comparable overall rating", f"{row['overall_rating']:.0f} / 5")
            st.caption("Selected issue theme observed in saved evidence." if row.get("theme_group") == 1 else "Saved issue evidence accompanies this result; absence of a theme does not establish reliability.")
            with st.expander("Supporting evidence"):
                if row.get("specification_cluster"):
                    st.caption("Specification subgroup: " + row["specification_cluster"])
                for item in json.loads(row.get("issue_evidence") or "[]"):
                    st.caption(f"{item['source']}: {item['record_count']} saved records; {item['usable_text_count']} usable narratives.")
                for item in json.loads(row.get("text_evidence") or "[]"):
                    if item["representation"] != "embeddings":
                        continue
                    st.markdown("**" + item["channel"].capitalize() + " themes**")
                    shares = [{"theme": c.removesuffix("_share"), "share": value} for c, value in item.items() if c.startswith("theme_") and c.endswith("_share")]
                    if shares:
                        st.altair_chart(alt.Chart(pd.DataFrame(shares)).mark_bar().encode(x=alt.X("share:Q", scale=alt.Scale(domain=[0, 1])), y="theme:N", tooltip=["theme", "share"]),
                                        alt="Shares of saved source records in semantic issue groups")
                    examples = item.get("representative_records", [])
                    for example in examples[:1]:
                        st.caption(example["record_id"] + " · " + example["evidence_date"][:10])
                        st.write(example["text"])
    if result["unranked_insufficient_issue_evidence"]:
        st.info(f"{result['unranked_insufficient_issue_evidence']:,} eligible listings have insufficient issue evidence for the requested avoidance comparison. They remain available through explicit browsing sorts.")
    st.download_button("Download recommendation evidence", json.dumps(result, indent=2, default=str), "recommendation_evidence.json", mime="application/json")
