"""Native Streamlit views for actuals, frozen evaluations, and source inspection."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math

import numpy as np

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st

from car_app.data import COHORT, FAMILY, ROOT, Filters, cached_evidence, filter_options, query_snapshot
from car_app.artifacts import (
    DEFAULT_BASE, DEFAULT_ENRICHED, errors, evaluation, inventory,
    paired_comparison, saved_shap, stat_key,
)
from car_app.state import update_dashboard_scope


def table(data, label, height=None):
    st.dataframe(data, hide_index=True, width="stretch", height=height or "auto", alt=label)


def download(data, label, filename, key):
    csv = data.write_csv() if isinstance(data, pl.DataFrame) else data.to_csv(index=False)
    st.download_button(label, csv, file_name=filename, mime="text/csv", key=key)


def money(value):
    return f"$ {value:,.0f}" if value is not None and math.isfinite(value) else "Unavailable"


def reset_filters():
    for key in list(st.session_state):
        if key.startswith("filter_"):
            del st.session_state[key]


def sidebar(source):
    with st.sidebar:
        st.header("Vehicle filters")
        st.caption("Dashboard scope · Empty dropdowns mean All")
        st.button("Reset filters", on_click=reset_filters, icon=":material/restart_alt:")
        with st.spinner("Loading vehicle choices…"):
            cohorts, categories = filter_options(source.identity)
        selections = {}
        scoped = cohorts
        for label, column, key in [
            ("Make", "Make", "makes"), ("Model", "Model", "models"),
            ("Model year", "ModelYear", "years"), ("Trim", "Trim", "trims"),
        ]:
            options = scoped[column].drop_nulls().unique().sort().to_list()
            options = [value for value in options if not isinstance(value, str) or value.strip()]
            widget = f"filter_{key}"
            if widget in st.session_state:
                st.session_state[widget] = [v for v in st.session_state[widget] if v in options]
            selected = st.multiselect(label, options, key=widget, placeholder="All")
            selections[key] = tuple(selected)
            if selected:
                scoped = scoped.filter(pl.col(column).is_in(selected))
        with st.expander("Price and attributes", expanded=False):
            for label, column, key in [
                ("Body style", "body_type", "body_types"), ("Fuel type", "fuel_type", "fuels"),
                ("Drivetrain", "wheel_system", "drivetrains"),
            ]:
                selections[key] = tuple(st.multiselect(label, sorted(categories[column]),
                                                       key=f"filter_{key}", placeholder="All"))
            selections["condition"] = st.selectbox("Condition", ["All", "New", "Used"], key="filter_condition")
            selections["min_price"] = st.number_input("Minimum advertised price (USD)", min_value=0.,
                                                       value=None, step=1000., key="filter_min_price")
            selections["max_price"] = st.number_input("Maximum advertised price (USD)", min_value=0.,
                                                       value=None, step=1000., key="filter_max_price")
            selections["max_mileage"] = st.number_input("Maximum advertised mileage", min_value=0.,
                                                         value=None, step=10000., key="filter_max_mileage")
            st.caption("Mileage units remain unverified in the source definitions.")
        filters = Filters(**selections)
        if filters.min_price is not None and filters.max_price is not None and filters.min_price > filters.max_price:
            st.warning("Minimum price exceeds maximum price; no listings will match.")
        update_dashboard_scope(st.session_state, filters, source.identity)
        st.caption(f"Snapshot {source.identity[:12]}")
        st.caption("Prices and availability are historical; this is not current inventory.")


def overview(source, filters):
    with st.spinner("Summarizing matching listings…"):
        summary = query_snapshot(source.identity, filters, "summary")
    with st.container(horizontal=True):
        for label, value in [
            ("Matching listings", f"{summary['listings']:,}"),
            ("Complete cohorts", f"{summary['cohorts']:,}"),
            ("Median advertised price", money(summary["median_price"])),
            ("Median advertised mileage", f"{summary['median_mileage']:,.0f}" if summary["median_mileage"] is not None else "Unavailable"),
        ]:
            st.metric(label, value, border=True)
    if not summary["listings"]:
        st.info("No listings match these filters. Adjust the sidebar or reset filters.")
        return
    st.caption(
        f"{summary['decoded_listings']:,} listings have a vPIC match; "
        f"{summary['complete_cohort_listings']:,} have all four decoded cohort fields. "
        f"Listing dates: {summary['first_listing']} to {summary['last_listing']}. Price-observation dates are unverified."
    )
    left, right = st.columns(2)
    for panel, column, width, maximum, title in [
        (left, "price", 5000, 200000, "Advertised price distribution"),
        (right, "mileage", 10000, 200000, "Advertised mileage distribution"),
    ]:
        with panel, st.container(border=True):
            st.subheader(title)
            cap = st.number_input("Chart upper bound" + (" (USD)" if column == "price" else ""),
                                  min_value=float(width), value=float(maximum), step=float(width),
                                  key=f"chart_bound_{column}")
            distribution = query_snapshot(source.identity, filters, "distribution", column, width)
            shown = distribution.filter(pl.col("lower") < cap)
            excluded = distribution.filter(pl.col("lower") >= cap)["listings"].sum()
            chart = alt.Chart(shown.to_pandas()).mark_bar(color="#176B87", orient="vertical").encode(
                x=alt.X("lower:Q", title="Advertised USD price" if column == "price" else "Advertised mileage"),
                x2="upper:Q", y=alt.Y("listings:Q", title="Listings"), y2=alt.datum(0),
                tooltip=["lower:Q", "upper:Q", "listings:Q"],
            ).properties(height=250)
            st.altair_chart(chart, alt=title)
            st.caption(f"Chart zoom only; {excluded:,} listings above the displayed bins remain in all statistics.")
    with st.container(border=True):
        st.subheader("Price and mileage")
        sample = query_snapshot(source.identity, filters, "scatter")
        chart = alt.Chart(sample.to_pandas()).mark_circle(size=18, opacity=.45).encode(
            x=alt.X("mileage:Q", title="Advertised mileage"),
            y=alt.Y("price:Q", title="Advertised price (USD)"),
            color=alt.Color("Make:N", legend=None), tooltip=[*COHORT, "price", "mileage"],
        ).properties(height=300).interactive()
        st.altair_chart(chart, alt="Advertised price versus mileage in a deterministic listing sample")
        st.caption(f"Deterministic VIN-hash sample, capped at 3,000 rows; {sample.height:,} points. Summary statistics use all matching rows.")
    st.subheader("Largest matching cohorts")
    cohorts = query_snapshot(source.identity, filters, "cohorts", 20)
    table(cohorts, "Twenty largest complete vehicle cohorts with observed price and mileage distributions")
    st.caption("Cohort = separate decoded model year, make, model, trim. Quartiles describe this filtered sample.")
    download(cohorts, "Download displayed cohorts", "cohorts.csv", "overview_cohorts_csv")


def listings(source, filters):
    mode = st.segmented_control("Explore records", ["Listings", "Cohorts"], default="Listings", key="record_mode")
    if mode == "Cohorts":
        cohorts = query_snapshot(source.identity, filters, "cohorts", 200)
        table(cohorts, "Largest 200 complete cohorts", 550)
        st.caption("Up to 200 largest complete cohorts; missing decoded identity remains visible in individual listings.")
        download(cohorts, "Download displayed cohorts", "cohorts.csv", "cohorts_csv")
        return
    summary = query_snapshot(source.identity, filters, "summary")
    with st.container(horizontal=True):
        sort = st.selectbox("Sort listings", ["price", "mileage", "ModelYear", "listing_id"])
        descending = st.checkbox("Descending")
        pages = max(1, math.ceil(summary["listings"] / 100))
        if st.session_state.get("listing_page", 1) > pages:
            st.session_state["listing_page"] = 1
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="listing_page")
    if page > pages:
        page = 1
    data = query_snapshot(source.identity, filters, "listings", sort, descending, page - 1)
    st.caption(f"{summary['listings']:,} matching listings · 100 rows per page · Missing decoded fields are not filled from listing labels.")
    table(data, "Filtered listing page with decoded identity, specifications, and source attributes", 570)
    download(data, "Download this page", "listing_page.csv", "listings_csv")
    with st.expander("Specifications and history definitions"):
        st.markdown(
            "Advertised body style, fuel, drivetrain, seating, horsepower, and equipment describe source listings. "
            "Reported accident/title fields are not independently verified. Blank values remain unknown. "
            "Fuel-economy measures are separate source fields; no combined MPG/MPGe score is calculated."
        )


def nhtsa(source, filters):
    st.caption("Distinct families represented by matching listings · Provisional complaint/recall cutoff: December 31, 2020")
    with st.spinner("Reading applicable NHTSA evidence…"):
        bundle = cached_evidence(source.identity, filters)
    with st.container(horizontal=True):
        for name, label in [("NHTSA_Complaints", "Saved eligible complaint IDs"),
                            ("NHTSA_Recalls", "Saved eligible recall campaigns"),
                            ("NHTSA_Safety", "Saved safety variants")]:
            st.metric(label, f"{bundle[name]['count']:,}", border=True)
    st.info("Counts describe saved reports and campaigns, not failure rates or unresolved VIN recalls. Missing source rows mean unknown coverage.")
    view = st.segmented_control("Evidence", ["Complaints", "Recalls", "Safety"], default="Complaints", key="nhtsa_view")
    name = {"Complaints": "NHTSA_Complaints", "Recalls": "NHTSA_Recalls", "Safety": "NHTSA_Safety"}[view]
    item = bundle[name]
    st.caption(f"{item['families']:,} of {item['reference_families']:,} selected families have eligible source rows; {item['conflicts']:,} source IDs have differing saved versions.")
    if item["records"].is_empty():
        st.info("No eligible saved evidence is available for this selection.")
        return
    if view in ["Complaints", "Recalls"]:
        counts = item["components"]
        if not counts.is_empty():
            column = "components" if view == "Complaints" else "Component"
            chart = alt.Chart(counts.to_pandas()).mark_bar(color="#176B87").encode(
                y=alt.Y(f"{column}:N", sort="-x", title=None), x=alt.X("source_ids:Q", title="Distinct source IDs"),
                tooltip=[column, "source_ids"],
            ).properties(height=300)
            st.altair_chart(chart, alt=f"Distinct {view.lower()} by reported component")
            st.caption("The chart shows up to 20 components. One source ID may mention multiple components. Conflicting versions are excluded from this breakdown.")
    else:
        st.warning("Ratings apply to the displayed NHTSA variants, not automatically to the selected trim or VIN. No universal safety ranking is computed.")
        st.caption("Zero/Not Rated star ratings are treated as missing. Overall/frontal comparisons require comparable classes, weights, and rating eras.")
    table(item["records"], f"Up to 500 saved {view.lower()} evidence rows", 500)
    st.caption("Source detail is capped at 500 deduplicated rows; headline counts and breakdowns use all eligible saved rows.")
    download(item["records"], "Download displayed evidence", f"{view.lower()}_evidence.csv", "nhtsa_csv")


def youtube(source, filters):
    st.caption("Make-level experimental descriptors · Explicit-comment attribution · Cumulative cutoff: October 1, 2026")
    with st.spinner("Reading supported make features…"):
        data = query_snapshot(source.identity, filters, "youtube")
    if data.is_empty():
        st.info("No selected make has a saved feature row. This does not establish absence of discussion.")
        return
    supported = data.filter(~pl.col("yt_support_suppressed"))
    with st.container(horizontal=True):
        st.metric("Matched makes", str(data.height), border=True)
        st.metric("Makes with topic support", str(supported.height), border=True)
        st.metric("Recorded comments across matched makes", f"{data['yt_comment_count'].sum():,}", border=True)
    st.info("These observations describe makes, not selected years, models, or trims. Independent automotive accuracy is unmeasured; directional sentiment is unavailable.")
    fields = [c for c in data.columns if c.endswith("_mention_share")]
    if fields:
        topics = data.select("Make", *fields).unpivot(index="Make", variable_name="topic", value_name="share")
        topics = topics.with_columns(pl.col("topic").str.replace("^yt_", "").str.replace("_mention_share$", ""),
                                     (pl.col("share") * 100).alias("share_pct")).drop_nulls("share")
        if not topics.is_empty():
            chart = alt.Chart(topics.to_pandas()).mark_bar().encode(
                y=alt.Y("topic:N", title=None), x=alt.X("share_pct:Q", title="Video-balanced mention share (%)"),
                color="Make:N", yOffset="Make:N", tooltip=["Make", "topic", "share_pct"],
            ).properties(height=370)
            st.altair_chart(chart, alt="Supported make-level discussion topic shares")
    columns = ["Make", "yt_comment_count", "yt_eligible_count", "yt_video_count", "yt_support_suppressed", *fields]
    table(data.select(columns), "Make-level discussion support and topic shares", 400)
    st.caption("Topic support requires at least 20 eligible comments and five videos. Unsupported measurements stay null; themes can overlap.")
    download(data, "Download matched make features", "youtube_make_features.csv", "youtube_csv")


def models(source, filters):
    st.caption("Saved evaluations only · No training, live scoring, or AI calls")
    with st.container(horizontal=True):
        legacy = st.checkbox("Include legacy inspection", key="model_legacy")
        if st.button("Refresh run inventory", icon=":material/refresh:"):
            st.session_state["artifact_refresh"] += 1
    runs, ignored = inventory(st.session_state["artifact_refresh"])
    available = {run.name: run for run in runs if legacy or not run.legacy}
    if not available:
        st.info("No readable completed model reports are available.")
        return
    defaults = [name for name in [DEFAULT_BASE, DEFAULT_ENRICHED] if name in available]
    remembered = st.session_state.get("selected_model_runs", defaults)
    current = st.session_state.get("model_runs", remembered)
    st.session_state["model_runs"] = [name for name in current if name in available]
    selected = st.multiselect("Saved runs", list(available), key="model_runs")
    st.session_state["selected_model_runs"] = selected
    st.caption(f"{len(ignored)} running, incomplete, or unreadable runs excluded. Run choices remain stable until changed.")
    with st.expander("Selected run metadata and artifacts"):
        indexed = []
        for name in selected:
            run = available[name]
            meta = run.metadata
            indexed.append({
                "run": name, "status": meta.get("status", "unrecorded"),
                "feature_set": meta.get("model_type", ", ".join(meta.get("feature_sets", []))),
                "estimators": ", ".join(run.table("metrics.csv").get("model",
                    run.table("validation_metrics.csv").get("model", pd.Series(dtype=str))).drop_duplicates()),
                "weighting": ", ".join(map(str, meta.get("price_weightings", ["unrecorded"]))),
                "split_strategy": meta.get("split_strategy", "unrecorded"),
                "validation_selected": meta.get("selected_by_validation_mae", meta.get("winner_selected_on", "unrecorded")),
                "test_receipts": len(list(run.path.glob("*_test_predictions.parquet"))),
                "saved_shap": len(list(run.path.glob("*/*_shap_global.csv"))),
                "legacy": run.legacy,
            })
        table(pd.DataFrame(indexed), "Selected run metadata and available artifact counts")
    if not selected:
        st.info("Select at least one saved run.")
        return
    mode = st.segmented_control("Evaluation", ["Test", "Validation"], default="Test", key="evaluation_mode")
    aggregate = []
    focused = []
    for name in selected:
        run = available[name]
        metrics = run.table("metrics.csv" if mode == "Test" else "validation_metrics.csv")
        if metrics.empty and mode == "Validation":
            test_report = run.table("metrics.csv")
            if "validation_mae" in test_report:
                metrics = test_report[["model", "validation_mae"]].rename(columns={"validation_mae": "mae"})
                st.caption(f"{name}: recorded selection-validation MAE only; other validation errors are unavailable.")
        if metrics.empty:
            st.info(f"{name}: no saved {mode.lower()} report.")
            continue
        overall = metrics.copy()
        overall.insert(0, "run", name)
        overall["scope"] = "overall saved " + mode.lower()
        aggregate.append(overall)
        if run.legacy:
            st.warning(f"{name}: legacy report; not eligible for automatic paired comparison or inference.")
        if mode == "Test":
            filtered = []
            for model in metrics.model.unique():
                file = run.prediction_path(model)
                if file is None:
                    filtered.append({"model": model, "scope": "filtered unavailable: no row-level receipt"})
                    continue
                try:
                    frame = evaluation(str(file), stat_key(file), source.identity, filters)
                    filtered.append({"model": model, "scope": "filtered test", **errors(frame)})
                except (ValueError, OSError) as exc:
                    filtered.append({"model": model, "scope": f"filtered unavailable: {exc}"})
            frame = pd.DataFrame(filtered)
            frame.insert(0, "run", name)
            focused.append(frame)
        elif filters.active():
            st.info(f"{name}: validation reports are aggregated; arbitrary dashboard-filtered validation metrics are unavailable.")
        if mode == "Validation":
            reports = [p.name for p in run.path.glob("validation_*.csv") if p.name != "validation_metrics.csv"]
            if reports:
                report = st.selectbox("Inspect aggregate validation report", reports, key=f"validation_report_{name}")
                saved = run.table(report)
                st.caption("Original aggregate report scope; dashboard filters are not applied or approximated.")
                table(saved, "Original saved validation report", 300)
                download(saved, "Download this aggregate report", report, f"report_csv_{name}")
    if not aggregate:
        return
    overall = pd.concat(aggregate, ignore_index=True)
    st.subheader("Overall saved performance")
    leading = [c for c in ["run", "variant", "model", "price_weighting", "scope", "mae", "rmse", "r2",
                           "wape_pct", "mean_signed_error", "p95_absolute_error", "test_rows",
                           "validation_mae", "validation_rmse", "validation_r2"] if c in overall]
    table(overall[leading], "Overall saved model performance", 360)
    download(overall, "Download overall metrics", "overall_metrics.csv", "overall_metrics_csv")
    st.caption("Model selection uses recorded validation MAE, not the best exposed test result. Bootstrap MAE intervals describe aggregate error uncertainty, not individual price intervals.")
    if focused:
        filtered = pd.concat(focused, ignore_index=True)
        st.subheader("Performance for dashboard filters")
        table(filtered, "Errors recomputed from matching saved test predictions", 360)
        download(filtered, "Download filtered metrics", "filtered_metrics.csv", "filtered_metrics_csv")
        if "mae" in filtered:
            plotted = filtered.dropna(subset=["mae"]).copy()
            if not plotted.empty:
                plotted["comparison"] = plotted["model"] + " / " + plotted["run"]
                chart = alt.Chart(plotted).mark_bar(color="#176B87").encode(
                    x=alt.X("mae:Q", title="Filtered test MAE (USD)"),
                    y=alt.Y("comparison:N", sort="x", title=None), tooltip=["run", "model", "rows", "mae", "support"])
                st.altair_chart(chart, alt="Filtered held-out MAE by saved run and estimator")
        st.caption("Low support means fewer than 100 evaluation rows or five vehicle families. Empty selections and undefined R² stay unavailable.")
    st.subheader("Prediction diagnostics and explanations")
    focus_name = st.selectbox("Diagnostic run", selected, key="diagnostic_run")
    focus = available[focus_name]
    test = focus.table("metrics.csv")
    if not test.empty and mode == "Test":
        model = st.selectbox("Estimator", test.model.unique().tolist(), key=f"diagnostic_model_{focus_name}")
        file = focus.prediction_path(model)
        if file:
            try:
                frame = evaluation(str(file), stat_key(file), source.identity, filters)
                if frame.empty:
                    st.info("No held-out predictions match these filters.")
                else:
                    diagnostics(frame)
            except (ValueError, OSError) as exc:
                st.warning(str(exc))
        shap = saved_shap(focus, model)
        if shap.empty:
            st.info("No saved SHAP explanation is available for this run and estimator.")
        else:
            scale = focus.metadata.get("shap_output_scale", "unrecorded; interpret this legacy scale cautiously")
            st.caption(f"Saved overall SHAP importance for this exact run; vehicle filters do not scope this explanation. Recorded output scale: {scale}.")
            table(shap.head(30), "Saved overall SHAP importance", 300)
    if DEFAULT_BASE in available and DEFAULT_ENRICHED in available and mode == "Test":
        if st.button("Compare pinned base/enriched runs on matching test rows"):
            try:
                with st.spinner("Checking fingerprints and paired evaluation identities…"):
                    paired = paired_comparison(available[DEFAULT_BASE], available[DEFAULT_ENRICHED], filters, source.identity)
                table(paired, "Paired base-versus-enriched errors for identical held-out listings")
                st.caption("Negative enriched-minus-base MAE means lower error. This comparison uses the pinned pair and active dashboard filters.")
                download(paired, "Download paired comparison", "paired_comparison.csv", "paired_csv")
            except (ValueError, OSError) as exc:
                st.warning(f"Paired comparison unavailable: {exc}")


def diagnostics(frame):
    sample = frame.sample(min(len(frame), 3000), random_state=42)
    left, right = st.columns(2)
    with left:
        scatter = alt.Chart(sample).mark_circle(size=18, opacity=.4).encode(
            x=alt.X("actual_price:Q", title="Observed advertised USD price"),
            y=alt.Y("prediction:Q", title="Predicted advertised USD price"),
            tooltip=["vin", *[c for c in COHORT if c in sample], "actual_price", "prediction"],
        ).properties(height=300)
        low = min(sample.actual_price.min(), sample.prediction.min())
        high = max(sample.actual_price.max(), sample.prediction.max())
        diagonal = alt.Chart(pd.DataFrame({"x": [low, high], "y": [low, high]})).mark_line(color="#555", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        st.altair_chart(scatter + diagonal, alt="Observed versus predicted prices with a perfect-prediction reference line")
    with right:
        residuals = frame.prediction - frame.actual_price
        counts, edges = np.histogram(residuals, bins=40)
        histogram = pd.DataFrame({"lower": edges[:-1], "upper": edges[1:], "rows": counts})
        chart = alt.Chart(histogram).mark_bar(color="#176B87", orient="vertical").encode(
            x=alt.X("lower:Q", title="Predicted minus observed price (USD)"), x2="upper:Q", y="rows:Q", y2=alt.datum(0)
        ).properties(height=300)
        st.altair_chart(chart, alt="Residual distribution using every filtered evaluation row")
    st.caption(f"Scatter: deterministic sample of {len(sample):,}; errors and residual histogram: all {len(frame):,} filtered evaluation rows.")
    dimension = st.selectbox("Error breakdown", [c for c in ["Make", "ModelYear", "is_new", "price_band"] if c in frame or c == "price_band"])
    grouped = frame.copy()
    if dimension == "price_band":
        grouped["price_band"] = (grouped.actual_price // 25000 * 25000).map(lambda n: f"{int(n):,}–{int(n + 25000):,}")
    rows = []
    for value, part in grouped.groupby(dimension, dropna=False):
        rows.append({dimension: str(value), **errors(part)})
    table(pd.DataFrame(rows), "Filtered model errors by selected segment", 350)


def source_inspection(source):
    st.caption("Snapshot-wide schema and profiles · Metadata remains fixed when vehicle filters change")
    records = []
    for key, entry in source.manifest["tables"].items():
        records.append({"table": key, "rows": entry["rows"], "columns": len(entry["schema"]),
                        "file": entry["file"], "size_mib": round(entry["bytes"] / 1024**2, 2)})
    table(pd.DataFrame(records), "Documented six-table snapshot inventory")
    options = list(source.manifest["tables"])
    selected = st.selectbox("Source table", options)
    schema = source.manifest["tables"][selected]["schema"]
    search = st.text_input("Find a field", key="dictionary_search")
    dictionary = (ROOT / "DATA_DICTIONARY.md").read_text(encoding="utf-8-sig")
    table_name = {"used_cars": "used_cars", "nhtsa_vpic_decodes": "NHTSA_vPIC",
                  "nhtsa_complaints": "NHTSA_Complaints", "nhtsa_recalls": "NHTSA_Recalls",
                  "nhtsa_safety_rating_values": "NHTSA_Safety", "youtube_comments_sentiment": "Youtube_Comments"}[selected]
    section = dictionary.split(f"## `{table_name}`", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 8:
            continue
        name = cells[0].strip("`")
        if search.lower() not in name.lower():
            continue
        rows.append({"field": name, "type": schema.get(name), "classification": cells[2],
                     "definition": cells[3], "missing": cells[4], "nonmissing": cells[5], "distinct": cells[6]})
    table(pd.DataFrame(rows), "Field definitions and snapshot-wide profiles", 550)
    st.caption("Definitions and profiles come from DATA_DICTIONARY.md; schemas come from manifest.json. Raw YouTube comments are unscored and are not attributed to individual dashboard cohorts.")
