"""Native Streamlit views for actuals, frozen evaluations, and source inspection."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import re

import numpy as np

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st

from car_app.data import COHORT, FAMILY, ROOT, Filters, cached_evidence, filter_options, query_snapshot
from car_app.artifacts import (
    diagnostic_inventory, errors, evaluation, evaluation_with_fallback,
    paired_comparison, saved_shap, stat_key,
)
from car_app.state import update_dashboard_scope


def table(data, label, height=None, column_config=None):
    fitted_height = min(height, 35 * (len(data) + 1) + 3) if height else "auto"
    st.dataframe(data, hide_index=True, width="stretch", height=fitted_height, row_height=35, column_config=column_config, alt=label)


def listing_display(data):
    hidden = {"listing_id", "listed_date", "make_name", "model_name", "trim_name",
              "year", "daysonmarket", "vpic_matched", "cohort_complete",
              "maximum_seating", "transmission"}
    first = [c for c in ["vin", *COHORT] if c in data.columns]
    remaining = [c for c in data.columns if c not in hidden and c not in first and c != "seats"]
    if "Transmission" in remaining and "seats" in data.columns:
        remaining.insert(remaining.index("Transmission") + 1, "seats")
    elif "seats" in data.columns:
        remaining.append("seats")
    return data.select(first + remaining)


def evidence_display(data, view):
    if view == "Complaints":
        data = data.drop([c for c in ["products", "odiNumber", "manufacturer"] if c in data.columns]).rename(
            {c: label for c, label in [("numberOfDeaths", "Deaths"), ("numberOfInjuries", "Injuries")]
             if c in data.columns})
        first = [c for c in ["dateComplaintFiled", "dateOfIncident", "vin", "model_year", "make", "model",
                            "components", "summary", "Deaths", "Injuries", "crash", "fire"] if c in data.columns]
        return data.select(first + [c for c in data.columns if c not in first])
    if view == "Recalls":
        data = data.drop([c for c in ["model_year", "make", "model"] if c in data.columns])
        columns = [c for c in data.columns if c not in ["ReportReceivedDate", "Remedy"]]
        if "Consequence" in columns and "Remedy" in data.columns:
            columns.insert(columns.index("Consequence") + 1, "Remedy")
        elif "Remedy" in data.columns:
            columns.append("Remedy")
        if "ReportReceivedDate" in data.columns:
            columns.insert(0, "ReportReceivedDate")
        return data.select(columns)
    if view == "Safety":
        preferred = ["ModelYear", "Make", "Model", "VehicleDescription", "ComplaintsCount",
                     "InvestigationCount", "OverallRating", "OverallCrashRating",
                     "OverallFrontCrashRating", "OverallSideCrashRating"]
        normalize = lambda name: re.sub(r"[^a-z0-9]", "", name.casefold())
        # Prefer canonical names, then keep the first remaining spelling of each field.
        kept, seen = [], {"vehicleid", "rolloverrating2", "rolloverpossibility2"}
        for column in [c for c in preferred if c in data.columns] + data.columns:
            key = normalize(column)
            if key not in seen:
                kept.append(column)
                seen.add(key)
        first = [c for c in preferred if c in kept]
        ratings = [c for c in kept if "rating" in normalize(c) and c not in first]
        return data.select(first + ratings + [c for c in kept if c not in first + ratings])
    return data


def youtube_display(data):
    renamed = data.rename({c: c.removeprefix("yt_") for c in data.columns if c.startswith("yt_")})
    identity = [c for c in ["Make", "support_suppressed"] if c in renamed.columns]
    fields = [c for c in renamed.columns if c not in identity]
    return (renamed.with_columns(pl.col(c).cast(pl.String).alias(c) for c in fields)
            .unpivot(index=identity, on=fields, variable_name="Metric", value_name="Value")
            .sort("Make", maintain_order=True))


def safety_long_display(records):
    identity = [c for c in ["ModelYear", "Make", "Model", "VehicleDescription"] if c in records.columns]
    fields = [c for c in records.columns if c not in identity]
    # Mixed counts, probabilities and text share a display-only string value column.
    return (records.with_columns(pl.col(c).cast(pl.String).alias(c) for c in fields)
            .unpivot(index=identity, on=fields, variable_name="Field", value_name="Value")
            .sort(identity, maintain_order=True))


def safety_rating_definitions(columns):
    """Explain displayed NCAP fields without assigning undocumented alias meanings."""
    descriptions = {
        "overallrating": "Overall Vehicle Score combines frontal, side and rollover injury risks.",
        "overallfrontcrashrating": "Combined driver and front-passenger protection in the frontal barrier test.",
        "frontcrashdriversiderating": "Driver protection in the frontal barrier test.",
        "frontcrashpassengersiderating": "Front-passenger protection in the frontal barrier test.",
        "overallsidecrashrating": "Overall side protection combines side-barrier and side-pole results.",
        "sidecrashdriversiderating": "Driver protection in the side-barrier test.",
        "sidecrashpassengersiderating": "Rear passenger protection in the side-barrier test (driver's side).",
        "sidepolecrashrating": "Driver protection in the side-pole test.",
        "sidebarrierratingoverall": "Combined occupant protection in the side-barrier test.",
        "combinedsidebarrierandpoleratingfront": "Front-seat side protection from combined barrier/pole assessment.",
        "combinedsidebarrierandpoleratingrear": "Rear-seat side protection in the combined side assessment; the pole test uses a front-seat dummy.",
        "rolloverrating": "Resistance to rollover in a single-vehicle crash; not occupant injury severity.",
    }
    overall_scale = "1–5 stars; 1 worst, 5 best (lower injury risk)."
    rows = []
    for column in columns:
        key = re.sub(r"[^a-z0-9]", "", column.casefold())
        if "rating" in key:
            description = descriptions.get(key)
            scale = "1–5 stars; 1 worst, 5 best (greater protection)."
            if key in {"overallrating", "overallfrontcrashrating", "overallsidecrashrating", "sidebarrierratingoverall"}:
                scale = overall_scale
            if key == "rolloverrating":
                scale = ("1–5 stars; 5 best. Risk: "
                         "1 >40%; 2 30–40%; 3 20–30%; 4 10–20%; 5 <10%.")
            if description is None:
                description = "Not defined in the saved dataset dictionary or verified NHTSA field guidance; no equivalence to OverallRating is assumed."
                scale = "Unverified; interpretation unavailable."
            rows.append({"Column": column, "Meaning": description, "Range and interpretation": scale})
        elif key == "rolloverpossibility":
            rows.append({"Column": column, "Meaning": "Estimated rollover probability conditional on a single-vehicle crash.",
                         "Range and interpretation": "0–1; lower is better (0.15 = 15%)."})
        elif key == "dynamictipresult":
            rows.append({"Column": column, "Meaning": "Whether the vehicle tipped up during the dynamic rollover maneuver.",
                         "Range and interpretation": "Tip / No Tip; No Tip is favorable."})
    return pd.DataFrame(rows)


def safety_rating_guide(columns):
    st.subheader("Safety ratings ranges / interpretations")
    st.table(safety_rating_definitions(columns), alt="Definitions and value ranges for displayed NHTSA safety ratings")
    st.caption("Blank = unavailable, including source 0/Not Rated. Compare within rating eras; "
               "frontal/overall scores also require the same class and weights within 250 lb.")
    st.markdown(
        "Sources: [NHTSA ratings and test descriptions](https://www.nhtsa.gov/ratings), "
        "[NCAP rating explanations](https://www.nhtsa.gov/ratings/resources-related-nhtsas-new-car-assessment-program), "
        "[rollover risk bands](https://www.ntsb.gov/safety/safety-studies/Documents/SR0203.pdf). "
        "Saved data originate from NHTSA `SafetyRatings/VehicleId/{vehicle_id}` after variant lookup by model year/make/model."
    )


def safety_media_items(records):
    columns = [c for c in records.columns if any(kind in c.casefold() for kind in ("picture", "video"))]
    items, seen = [], set()
    for row in records.iter_rows(named=True):
        description = row.get("VehicleDescription") or "Vehicle description unavailable"
        for column in columns:
            url = row[column]
            if not isinstance(url, str) or not url.strip().startswith(("https://", "http://")):
                continue
            item = (description, column, url.strip())
            if item not in seen:
                items.append(item)
                seen.add(item)
    items.sort(key=lambda item: "video" in item[1].casefold())
    return columns, items


def safety_media(items):
    st.subheader("Crash pictures and videos")
    if not items:
        st.info("No saved crash media are available for these vehicles.")
        return
    pages = math.ceil(len(items) / 6)
    if st.session_state.get("safety_media_page", 1) > pages:
        st.session_state["safety_media_page"] = 1
    page = st.number_input("Media page", min_value=1, max_value=pages, value=1,
                           step=1, key="safety_media_page")
    st.caption(f"{len(items):,} media items · Up to 6 per page. Older WMV videos may not play in your browser.")
    for description, column, url in items[(page - 1) * 6:page * 6]:
        with st.container(border=True):
            label = f"{column} · {description}"
            st.text(label)
            if "video" in column.casefold():
                st.video(url, alt=label)
            else:
                st.image(url, alt=label, width=500)


def download(data, label, filename, key):
    csv = data.write_csv() if isinstance(data, pl.DataFrame) else data.to_csv(index=False)
    st.download_button(label, csv, file_name=filename, mime="text/csv", key=key)


def money(value):
    return f"$ {value:,.0f}" if value is not None and math.isfinite(value) else "Unavailable"


def reset_filters():
    for key in list(st.session_state):
        if key.startswith("filter_"):
            del st.session_state[key]


def single_filter(label, options, key):
    current = st.session_state.get(key)
    if isinstance(current, (list, tuple)):
        st.session_state[key] = next((value for value in current if value in options), None)
    elif current is not None and current not in options:
        st.session_state[key] = None
    selected = st.selectbox(label, [None, *options], key=key,
                            format_func=lambda value: "All" if value is None else str(value))
    return () if selected is None else (selected,)


def sidebar(source):
    with st.sidebar:
        st.header("Vehicle filters")
        st.caption("Choose one value per dropdown, or All. Makes require at least 10 listing rows in the snapshot.")
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
            selected = single_filter(label, options, f"filter_{key}")
            selections[key] = selected
            if selected:
                scoped = scoped.filter(pl.col(column).is_in(selected))
        with st.expander("Price and attributes", expanded=False):
            for label, column, key in [
                ("Body style", "body_type", "body_types"), ("Fuel type", "fuel_type", "fuels"),
                ("Drivetrain", "wheel_system", "drivetrains"),
            ]:
                selections[key] = single_filter(label, sorted(categories[column]), f"filter_{key}")
            selections["condition"] = st.selectbox("Condition", ["All", "New", "Used"], key="filter_condition")
            selections["min_price"] = st.number_input("Minimum advertised price (USD)", min_value=0.,
                                                       value=None, step=1000., key="filter_min_price")
            selections["max_price"] = st.number_input("Maximum advertised price (USD)", min_value=0.,
                                                       value=None, step=1000., key="filter_max_price")
            selections["max_mileage"] = st.number_input("Maximum advertised mileage", min_value=0.,
                                                         value=None, step=10000., key="filter_max_mileage")
            st.caption("Mileage units remain unverified in the source definitions.")
        selections["makes"] = selections["makes"] or tuple(cohorts["Make"].drop_nulls().unique().sort().to_list()) or ("__no_supported_make__",)
        filters = Filters(**selections)
        if filters.min_price is not None and filters.max_price is not None and filters.min_price > filters.max_price:
            st.warning("Minimum price exceeds maximum price; no listings will match.")
        update_dashboard_scope(st.session_state, filters, source.identity)
        if any([filters.makes, filters.models, filters.years, filters.trims]):
            with st.expander("Matching make / model / year scope"):
                scope = query_snapshot(source.identity, filters, "scope")
                table(scope.head(200), "Matching vehicle families and listing counts")
                st.caption(f"{scope.height:,} matching families; up to 200 shown. Model/year/trim choices narrow the selected make. "
                           "Sorted listing pages and largest-cohort tables may show only part of this scope.")

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
    for panel, column, title in [
        (left, "price", "Advertised price distribution"),
        (right, "mileage", "Advertised mileage distribution"),
    ]:
        with panel, st.container(border=True):
            st.subheader(title)
            observed_min, observed_max = summary[f"min_{column}"], summary[f"max_{column}"]
            if observed_min is None or observed_max is None:
                st.info("No finite values match these filters.")
                continue
            lower = filters.min_price if column == "price" else None
            upper = filters.max_price if column == "price" else filters.max_mileage
            lower = float(observed_min if lower is None else lower)
            upper = float(observed_max if upper is None else upper)
            # A constant-valued selection still needs a nonzero plotting domain.
            if upper <= lower:
                lower, upper = max(0., lower - .5), upper + .5
            width = max(1, math.ceil((upper - lower) / 30))
            distribution = query_snapshot(source.identity, filters, "distribution", column, width)
            # Keep values exactly on the upper boundary in the final visible bin.
            shown = distribution.with_columns(
                pl.when(pl.col("lower") >= upper).then(upper - width)
                .otherwise(pl.col("lower")).alias("lower"),
                pl.when(pl.col("lower") >= upper).then(upper)
                .otherwise(pl.col("upper")).alias("upper"),
            ).group_by("lower", "upper").agg(pl.col("listings").sum()).with_columns(
                pl.col("lower").clip(lower_bound=lower),
                pl.col("upper").clip(upper_bound=upper))
            chart = alt.Chart(shown.to_pandas()).mark_bar(color="#176B87", orient="vertical", clip=True).encode(
                x=alt.X("lower:Q", title="Advertised USD price" if column == "price" else "Advertised mileage",
                        scale=alt.Scale(domain=[lower, upper], nice=False), axis=alt.Axis(format=",d")),
                x2="upper:Q", y=alt.Y("listings:Q", title="Listings"), y2=alt.datum(0),
                tooltip=["lower:Q", "upper:Q", "listings:Q"],
            ).properties(height=250)
            st.altair_chart(chart, alt=title)
            st.caption("Axis follows the active limits; unset bounds follow matching listings. All matching finite values are included.")
            label = "highest advertised prices" if column == "price" else "highest advertised mileages"
            with st.expander(f"Top 10 vehicles by {label}"):
                extremes = query_snapshot(source.identity, filters, "distribution_extremes", column)
                table(extremes, f"Ten filtered listings with the {label}", column_config={
                    "price": st.column_config.NumberColumn("Advertised price (USD)", format="dollar"),
                    "mileage": st.column_config.NumberColumn("Advertised mileage", format="localized"),
                })
                st.caption("Highest first within the active filters; one row per listing. Mileage uses source units. These rows remain in the chart and statistics.")
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
    data = listing_display(data)
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
    if view == "Safety":
        safety_rating_guide(evidence_display(item["records"], view).columns)
    st.caption(f"{item['families']:,} of {item['reference_families']:,} selected families have eligible source rows; {item['conflicts']:,} source IDs have differing saved versions.")
    if item["records"].is_empty():
        st.info("No eligible saved evidence is available for this selection.")
        return
    if view in ["Complaints", "Recalls"]:
        counts = item["components"]
        if not counts.is_empty():
            column = "components" if view == "Complaints" else "Component"
            chart = alt.Chart(counts.to_pandas()).mark_bar(color="#176B87").encode(
                y=alt.Y(f"{column}:N", sort="-x", title=None), x=alt.X("source_ids:Q", title="Distinct source IDs",
                    axis=alt.Axis(format=",d", tickMinStep=1)),
                tooltip=[column, alt.Tooltip("source_ids:Q", format=",d")],
            ).properties(height=300)
            st.altair_chart(chart, alt=f"Distinct {view.lower()} by reported component")
            st.caption("The chart shows up to 20 components. One source ID may mention multiple components. Conflicting versions are excluded from this breakdown.")
    else:
        st.warning("Ratings apply to the displayed NHTSA variants, not automatically to the selected trim or VIN. No universal safety ranking is computed.")
        st.caption("Zero/Not Rated star ratings are treated as missing. Overall/frontal comparisons require comparable classes, weights, and rating eras.")
    records = evidence_display(item["records"], view)
    media_items = []
    if view == "Safety":
        media_columns, media_items = safety_media_items(records)
        records = safety_long_display(records.drop(media_columns))
        st.caption("One row per vehicle and field; blank values remain unavailable.")
    table(records, f"Up to 500 saved {view.lower()} evidence rows", 500)
    st.caption("Source detail is capped at 500 deduplicated rows; headline counts and breakdowns use all eligible saved rows.")
    download(records, "Download displayed evidence", f"{view.lower()}_evidence.csv", "nhtsa_csv")
    if view == "Safety":
        safety_media(media_items)


def youtube(source, filters):
    st.caption("Make-level experimental descriptors · Explicit-comment attribution · Cumulative cutoff: October 1, 2026")
    with st.spinner("Reading supported make features…"):
        data = query_snapshot(source.identity, filters, "youtube")
    available = query_snapshot(source.identity, Filters(), "youtube")
    eligible_cohorts, _ = filter_options(source.identity)
    available = available.filter(pl.col("Make").is_in(eligible_cohorts["Make"].drop_nulls().unique().to_list())) if "Make" in available.columns else available
    base_makes = data["Make"].to_list() if "Make" in data.columns else []
    options = sorted(set(available["Make"].to_list()) - set(base_makes)) if "Make" in available.columns else []
    if st.session_state.get("youtube_compare_make") not in [None, *options]:
        st.session_state["youtube_compare_make"] = None
    comparison = st.selectbox("Compare with another make", [None, *options],
                              key="youtube_compare_make",
                              format_func=lambda value: "No comparison" if value is None else value)
    st.caption("Comparison applies only to YouTube. The added make uses make-level evidence, independent of sidebar model/year and listing attributes.")
    comparison_filters = Filters(makes=(comparison,)) if comparison else None
    if comparison_filters:
        extra = query_snapshot(source.identity, comparison_filters, "youtube")
        data = pl.concat([data, extra]).unique(subset="Make", maintain_order=True).sort("Make")
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
        topics = topics.with_columns(pl.col("topic").str.replace("^yt_", "").str.replace("_mention_share$", "").str.replace_all("_", " "),
                                     (pl.col("share") * 100).alias("share_pct")).drop_nulls("share")
        if not topics.is_empty():
            chart = alt.Chart(topics.to_pandas()).mark_bar().encode(
                y=alt.Y("topic:N", title=None), x=alt.X("share_pct:Q", title="Video-balanced mention share (%)"),
                color="Make:N", yOffset="Make:N", tooltip=["Make", "topic", "share_pct"],
            ).properties(height=370)
            st.altair_chart(chart, alt="Supported make-level discussion topic shares")
    columns = ["Make", "yt_comment_count", "yt_eligible_count", "yt_video_count", "yt_support_suppressed", *fields]
    displayed = youtube_display(data.select(columns))
    table(displayed, "Make-level discussion support and topic shares", 400)
    st.caption("Topic support requires at least 20 eligible comments and five videos. Unsupported measurements stay null; themes can overlap.")
    download(youtube_display(data), "Download matched make features", "youtube_make_features.csv", "youtube_csv")
    st.subheader("Example comments for selected makes")
    comments = query_snapshot(source.identity, filters, "youtube_comments")
    if comparison_filters:
        extra_comments = query_snapshot(source.identity, comparison_filters, "youtube_comments")
        comments = pl.concat([comments.head(5), extra_comments.head(5)], how="diagonal_relaxed")
    st.caption("Up to 10 saved comments with explicit make attribution and eligible scored text. "
               "Examples describe makes, not selected years, models, or trims; they are not a representative sample.")
    if comments.is_empty():
        st.info("No eligible saved comment examples are available for these makes.")
    else:
        for row in comments.iter_rows(named=True):
            with st.container(border=True):
                st.caption(f"{row['Make']} · {row['video_title']} · {row['published_date']}")
                st.text(row["text"])



def model_table(data, label, height=None):
    labels = {
        "feature_set": "Feature set", "model": "Model", "price_weighting": "Price weighting",
        "scope": "Evaluation scope", "requested_rows": "Rows matching selected filters", "rows": "Evaluated listings", "test_rows": "Test listings",
        "family_blocks": "Vehicle families", "support": "Sample support", "price_band": "Observed price band (USD)",
        "mae": "Mean absolute error (USD)", "rmse": "Root mean squared error (USD)",
        "mse": "Mean squared error (USD²)", "r2": "R² score",
        "wape_pct": "Weighted absolute percentage error (%)",
        "mean_signed_error": "Average prediction bias (USD)",
        "p95_absolute_error": "95th percentile absolute error (USD)",
        "base_mae": "Base mean absolute error (USD)",
        "enriched_mae": "Enhanced mean absolute error (USD)",
        "enriched_minus_base_mae": "Enhanced minus base error (USD)",
    }
    columns = {}
    for name in data.columns:
        if name.startswith("validation_"):
            metric = name.removeprefix("validation_")
            columns[name] = "Validation " + labels.get(metric, metric.replace("_", " ")).lower().replace("usd", "USD").replace("r²", "R²")
        else:
            columns[name] = labels.get(name, name.replace("_", " ").capitalize())
    table(data, label, height, column_config=columns)


def models(source, filters):
    st.header("Model performance and feature comparisons")
    st.caption("Saved evaluations only · No training, live scoring, or AI calls")
    if st.button("Refresh saved evaluations", icon=":material/refresh:"):
        st.session_state["artifact_refresh"] += 1
    try:
        available, candidates, model_status, inspected = diagnostic_inventory()
    except (ValueError, OSError) as exc:
        st.warning(f"Checkpoint discovery unavailable: {exc}")
        return
    if not available:
        st.info("No completed model evaluations are available.")
        return
    selected = list(available)
    st.caption("Runs may use different data or training protocols; displayed metrics do not establish a matched experiment.")
    st.caption("Newest completed full-data checkpoint per feature-set/model combination. Discovery refreshes on every interaction.")
    mode = st.segmented_control("Evaluation", ["Test", "Validation"], default="Test", key="evaluation_mode")
    st.caption("Validation predictions evaluate settings selected during development; test predictions evaluate the final refit on untouched families. The same vehicle filters apply separately to each partition.")
    aggregate = []
    focused = []
    for name in selected:
        run = available[name]
        metrics = run.table("metrics.csv" if mode == "Test" else "validation_metrics.csv")
        allowed_models = candidates.loc[candidates.run.eq(name), "model"]
        if "model" in metrics:
            metrics = metrics[metrics.model.isin(allowed_models)]
        if metrics.empty and mode == "Validation":
            test_report = run.table("metrics.csv")
            if "validation_mae" in test_report:
                metrics = test_report[["model", "validation_mae"]].rename(columns={"validation_mae": "mae"})
                st.caption(f"{run.metadata['model_type']}: recorded selection-validation MAE only; other validation errors are unavailable.")
        if metrics.empty:
            st.info(f"{run.metadata['model_type']}: no saved {mode.lower()} report.")
            continue
        overall = metrics.copy()
        overall.insert(0, "feature_set", run.metadata["model_type"])
        overall["scope"] = "overall saved " + mode.lower()
        aggregate.append(overall)
        if run.legacy:
            st.warning(f"{run.metadata['model_type']}: legacy report; not eligible for automatic paired comparison or inference.")
        filtered = []
        for model in metrics.model.unique():
            file = run.prediction_path(model, mode.lower())
            if file is None:
                continue
            try:
                frame, scope, requested_rows = evaluation_with_fallback(str(file), stat_key(file), source.identity, filters)
                filtered.append({"model": model, "scope": scope, "requested_rows": requested_rows, **errors(frame)})
            except (ValueError, OSError) as exc:
                filtered.append({"model": model, "scope": f"filtered unavailable: {exc}"})
        if filtered:
            frame = pd.DataFrame(filtered)
            frame.insert(0, "feature_set", run.metadata["model_type"])
            focused.append(frame)
    if not aggregate:
        return
    overall = pd.concat(aggregate, ignore_index=True)
    st.subheader("Overall saved performance")
    leading = [c for c in ["feature_set", "model", "price_weighting", "scope", "mae", "rmse", "r2",
                           "wape_pct", "mean_signed_error", "p95_absolute_error", "test_rows",
                           "validation_mae", "validation_rmse", "validation_r2"] if c in overall]
    model_table(overall[leading], "Overall saved model performance", 360)
    download(overall, "Download overall metrics", "overall_metrics.csv", "overall_metrics_csv")
    st.caption("Model selection uses recorded validation MAE, not the best exposed test result. Bootstrap MAE intervals describe aggregate error uncertainty, not individual price intervals.")
    if focused:
        filtered = pd.concat(focused, ignore_index=True)
        st.subheader("Performance for dashboard filters")
        st.warning("Year/make/model families are held out together. Narrow filters can match few or no evaluation rows. With fewer than 100 rows, this section broadens to make/model, then make. Each row identifies its actual scope; sidebar filters stay unchanged. Broader scopes drop other vehicle filters.")
        model_table(filtered, "Errors recomputed from matching saved evaluation predictions", 360)
        download(filtered, "Download filtered metrics", "filtered_metrics.csv", "filtered_metrics_csv")
        if "mae" in filtered:
            plotted = filtered.dropna(subset=["mae"]).copy()
            if not plotted.empty:
                chart = alt.Chart(plotted).mark_bar().encode(
                    x=alt.X("mae:Q", title=f"Filtered {mode.lower()} MAE (USD)"),
                    y=alt.Y("model:N", title="Model", sort=["catboost", "lightgbm", "xgboost", "random_forest", "multiple_linear_regression", "lasso", "ridge", "elastic_net", "pcr"]),
                    yOffset=alt.YOffset("feature_set:N", sort=["base", "enhanced", "enhanced_safety", "full"]),
                    color=alt.Color("feature_set:N", title="Feature set", sort=["base", "enhanced", "enhanced_safety", "full"],
                                    scale=alt.Scale(domain=["base", "enhanced", "enhanced_safety", "full"], range=["#315E88", "#A77925", "#9A4C24", "#74618C"])), tooltip=["feature_set", "model", "scope", "rows", "mae", "support"])
                st.altair_chart(chart, alt="Filtered held-out MAE by feature set and estimator")

        st.caption("Low support means fewer than 100 evaluation rows or five vehicle families. Empty selections and undefined R² stay unavailable.")
    st.subheader("Prediction diagnostics and feature importance")
    feature_set = st.selectbox("Diagnostic feature set", candidates.feature_set.drop_duplicates().tolist(), key="diagnostic_feature_set")
    choices = candidates[candidates.feature_set.eq(feature_set)]
    model = st.selectbox("Estimator", choices.model.tolist(), key=f"diagnostic_estimator_{feature_set}")
    focus_name = choices.loc[choices.model.eq(model), "run"].iloc[0]
    focus = available[focus_name]
    file = focus.prediction_path(model, mode.lower())
    if file:
        try:
            frame, scope, requested_rows = evaluation_with_fallback(str(file), stat_key(file), source.identity, filters)
            st.caption(f"Diagnostic scope: {scope}; {len(frame):,} rows ({requested_rows:,} matched the original filters).")
            if frame.empty:
                st.info("No saved evaluation predictions match these filters.")
            else:
                diagnostics(frame)
        except (ValueError, OSError) as exc:
            st.warning(str(exc))
    shap = saved_shap(focus, model)
    if shap.empty:
        st.info("No saved notebook explanation is available for this estimator.")
    else:
        st.caption(f"{shap.attrs.get('method', 'Saved SHAP importance')}. Scale: {shap.attrs.get('scale', focus.metadata.get('shap_output_scale', 'unrecorded'))}. Overall explanation; vehicle filters do not scope it.")
        importance = shap.head(10).copy()
        value_column = "value" if "value" in importance else next((c for c in importance if c != "feature" and pd.api.types.is_numeric_dtype(importance[c])), None)
        if value_column:
            importance["magnitude"] = importance[value_column].abs()
            signed = "coefficient" in shap.attrs.get("method", "").lower()
            chart = alt.Chart(importance).mark_bar().encode(
                x=alt.X(f"{value_column}:Q", title="Signed coefficient" if signed else "Feature importance", scale=alt.Scale(zero=True)),
                y=alt.Y("feature:N", sort=alt.SortField(field="magnitude", order="descending"), title=None),
                color=alt.condition(alt.datum[value_column] < 0, alt.value("#9A4C24"), alt.value("#315E88")),
                tooltip=["feature", value_column],
            ).properties(height=320)
            st.altair_chart(chart, alt="Top 10 model features by importance magnitude; coefficients retain their signs")
    base_name = next((name for name in selected if name.startswith("base/")), None)
    enhanced_name = next((name for name in selected if name.startswith("enhanced/")), None)
    if base_name and enhanced_name and mode == "Test":
        if st.button("Compare base/enhanced on matching test rows"):
            try:
                with st.spinner("Checking fingerprints and paired evaluation identities…"):
                    paired = paired_comparison(available[base_name], available[enhanced_name], filters, source.identity)
                model_table(paired, "Paired base-versus-enriched errors for identical held-out listings")
                st.caption("Negative enriched-minus-base MAE means lower error. This comparison uses the latest selected pair and active dashboard filters.")
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
        view = st.segmented_control("Error distribution range", ["Middle 98%", "Full range"],
                                    default="Middle 98%", key="residual_range")
        shown = residuals
        if view == "Middle 98%" and len(residuals) > 1:
            lower, upper = residuals.quantile([0.01, 0.99])
            shown = residuals[residuals.between(lower, upper)]
        counts, edges = np.histogram(shown, bins=40)
        histogram = pd.DataFrame({"lower": edges[:-1], "upper": edges[1:], "rows": counts})
        bars = alt.Chart(histogram).mark_bar(color="#176B87", orient="vertical").encode(
            x=alt.X("lower:Q", title="Predicted minus observed price (USD)"), x2="upper:Q",
            y=alt.Y("rows:Q", title="Number of listings"), y2=alt.datum(0),
            tooltip=[alt.Tooltip("lower:Q", title="Error from ($)", format=",.0f"),
                     alt.Tooltip("upper:Q", title="Error to ($)", format=",.0f"), "rows:Q"],
        )
        zero = alt.Chart(pd.DataFrame({"error": [0]})).mark_rule(color="#9A4C24").encode(x="error:Q")
        st.altair_chart((bars + zero).properties(height=300), alt="Distribution of prediction errors; zero means an exact prediction")
        st.caption(f"Showing {len(shown):,} of {len(residuals):,} errors; {len(residuals) - len(shown):,} outside this view. Metrics include every evaluation row.")
    st.caption(f"Scatter: deterministic sample of {len(sample):,}. Negative errors mean underprediction; positive errors mean overprediction. Each histogram bar counts listings in an error range, not makes or models.")
    dimension = st.selectbox("Error breakdown", [c for c in ["Make", "price_band"] if c in frame or c == "price_band"],
                             format_func=lambda value: "Price band" if value == "price_band" else value)
    grouped = frame.copy()
    if dimension == "price_band":
        grouped["price_band"] = (grouped.actual_price // 25000 * 25000).map(lambda n: f"{int(n):,}–{int(n + 25000):,}")
    rows = []
    for value, part in grouped.groupby(dimension, dropna=False):
        rows.append({dimension: str(value), **errors(part)})
    model_table(pd.DataFrame(rows), "Filtered model errors by selected segment", 350)


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
