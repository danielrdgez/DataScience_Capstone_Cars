"""Native Streamlit views for actuals, frozen evaluations, and source inspection."""
from __future__ import annotations

import math
import re

import numpy as np

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st

from car_app.data import COHORT, ROOT, Filters, cached_evidence, filter_options, query_snapshot
from car_app.artifacts import (
    diagnostic_inventory, errors, evaluation_with_fallback,
    paired_comparison, saved_shap, stat_key, youtube_price_contract,
)
from car_app.state import update_dashboard_scope


def table(data, label, height=None, column_config=None):
    fitted_height = min(height, 35 * (len(data) + 1) + 3) if height else "auto"
    st.dataframe(data, hide_index=True, width="stretch", height=fitted_height, row_height=35, column_config=column_config, alt=label)


def listing_display(data):
    hidden = {"listing_id", "listed_date", "make_name", "model_name", "trim_name",
              "year", "daysonmarket", "vpic_matched", "cohort_complete",
              "maximum_seating", "transmission", "city_fuel_economy",
              "highway_fuel_economy", "has_accidents", "frame_damaged",
              "salvage", "theft_title", "major_options", "city"}
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


def youtube_feature_definition(field):
    name = field.removeprefix("yt_")
    counts = {"comment_count": "Deduplicated attributable comments.",
              "eligible_count": "Comments with eligible vehicle content.",
              "video_count": "Videos with eligible vehicle content.",
              "feature_row_present": "Whether a matching make feature row exists."}
    if name in counts:
        return "Evidence support", counts[name]
    if name.startswith("semantic_") and name != "semantic_missing":
        return "Semantic coordinates", "Training-fitted PCA coordinate of make-masked text embeddings; no fixed range or standalone topic meaning."
    if name.endswith("_mention_share"):
        topic = name.removesuffix("_mention_share").replace("_", " ")
        return "Discussion topics", f"Video-balanced share of eligible comments mentioning {topic} (0–1); topics can overlap."
    if name.endswith("missing") or name == "support_suppressed":
        scope = name.removesuffix("_missing").replace("_", " ")
        meaning = ("Insufficient semantic/topic evidence: fewer than 20 eligible comments or five videos."
                   if name in {"semantic_missing", "support_suppressed"} or name.endswith("_topic_missing")
                   else f"Insufficient {scope} sentiment evidence: fewer than 20 resolved opinions or five opinion videos.")
        return "Evidence support", meaning + " Yes means insufficient support."
    measurements = {
        "log1p_opinion_video_count": "Natural log of one plus contributing opinion-video count.",
        "log1p_opinion_count": "Natural log of one plus resolved-opinion comment count.",
        "opinion_video_count": "Contributing videos with resolved opinions; retained below score support.",
        "opinion_count": "Resolved-opinion comments; retained below score support.",
        "video_balanced_soft_net": "Video-balanced positive minus negative tone probabilities (−1 to 1).",
        "video_balanced_net": "Video-balanced (positive − negative) / all resolved opinions, including neutral and mixed (−1 to 1).",
        "mean_confidence": "Mean winning tone probability (0–1); not validated correctness.",
        "mean_margin": "Mean lead of the winning tone probability over the runner-up (0–1).",
        "mean_entropy": "Mean three-class tone entropy, divided by ln(3) (0–1); higher means less concentrated probabilities.",
        "low_confidence_share": "Video-balanced share of opinions with a clause below 0.75 confidence or 0.15 margin (0–1).",
    }
    for suffix, meaning in measurements.items():
        if name.endswith("_" + suffix):
            scope = name.removesuffix("_" + suffix).replace("_", " ")
            group = ("Overall sentiment" if scope == "overall" else
                     "Evidence support" if "count" in suffix else "Aspect sentiment and uncertainty")
            return group, f"{scope.capitalize()}: {meaning}"
    raise ValueError("No YouTube feature definition: " + field)


def youtube_display(data, columns=None):
    from MODELS.Price_ML_Models import YOUTUBE_LOG_COUNT_COLUMNS

    sources = {target: source for source, target in YOUTUBE_LOG_COUNT_COLUMNS.items()}
    columns = [c for c in data.columns if c != "Make"] if columns is None else columns
    definitions = {field: youtube_feature_definition(field) for field in columns}
    rows = []
    for record in data.iter_rows(named=True):
        for field in columns:
            value = record[field]
            group, meaning = definitions[field]
            missing = value is None or (isinstance(value, float) and not math.isfinite(value))
            flag = field.endswith("_missing") or field == "yt_support_suppressed"
            evidence = ("Unavailable / insufficient evidence" if missing else
                        "Insufficient support" if flag and value else "Support met" if flag else
                        "Observed support" if "count" in field or field == "yt_feature_row_present" else "Supported")
            displayed = (None if missing else "Yes" if isinstance(value, bool) and value else
                         "No" if isinstance(value, bool) else str(value))
            rows.append({"Group": group, "Feature": field, "Meaning": meaning, "Make": record["Make"],
                         "Value": displayed, "Transformation": "ln(1 + count)" if field in sources else "As exported",
                         "Original count": str(record[sources[field]]) if field in sources and record[sources[field]] is not None else None,
                         "Evidence": evidence})
    return pl.DataFrame(rows, schema={c: pl.String for c in (
        "Group", "Feature", "Meaning", "Make", "Value", "Transformation", "Original count", "Evidence")}).sort("Group", "Feature", "Make")


def youtube_feature_tables(data, columns):
    from MODELS.Price_ML_Models import YOUTUBE_LOG_COUNT_COLUMNS
    from NHTSA.text_features import join_youtube_features

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate YouTube inputs in the pricing receipt")
    model_values = join_youtube_features(data.select("Make"), data)
    if not set(columns).issubset(model_values.columns):
        raise ValueError("YouTube export does not supply all recorded pricing inputs")
    # Raw counts already accompany their transformed inputs in the main table.
    represented = {*columns, *[source for source, target in YOUTUBE_LOG_COUNT_COLUMNS.items() if target in columns]}
    additional = [c for c in data.columns if c != "Make" and c not in represented]
    return youtube_display(model_values, columns), youtube_display(data, additional)


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
    st.divider()
    st.subheader("Evidence patterns and comparisons")
    from car_app.discovery import nhtsa_dashboard
    try:
        nhtsa_dashboard(source, filters, view=view)
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Saved discovery charts unavailable: {exc}")


def youtube(source, filters):
    try:
        contract = youtube_price_contract()
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"YouTube pricing inputs unavailable: {exc}")
        return
    feature_args = (contract["directory"], contract["source_signature"])
    st.caption(f"Make-level experimental descriptors · Explicit-comment attribution · Cumulative cutoff: {contract['as_of']}")
    available = query_snapshot(source.identity, Filters(), "youtube", *feature_args)
    eligible_cohorts, _ = filter_options(source.identity)
    available = available.filter(pl.col("Make").is_in(eligible_cohorts["Make"].drop_nulls().unique().to_list())) if "Make" in available.columns else available
    makes = sorted(available["Make"].drop_nulls().unique().to_list()) if "Make" in available.columns else []
    if not makes:
        st.info("No make has a saved feature row. This does not establish absence of discussion.")
        return
    if st.session_state.get("youtube_base_make") not in makes:
        st.session_state["youtube_base_make"] = makes[0]
    base_make = st.selectbox("YouTube make", makes, key="youtube_base_make")
    st.caption("YouTube has its own make selection, starting with the first make alphabetically. "
               "These charts and comments are independent of the dashboard sidebar filters.")
    youtube_filters = Filters(makes=(base_make,))
    data = available.filter(pl.col("Make") == base_make)
    options = [make for make in makes if make != base_make]
    if st.session_state.get("youtube_compare_make") not in [None, *options]:
        st.session_state["youtube_compare_make"] = None
    comparison = st.selectbox("Compare with another make", [None, *options],
                              key="youtube_compare_make",
                              format_func=lambda value: "No comparison" if value is None else value)
    st.caption("Optional comparison applies to the two YouTube charts.")
    comparison_filters = Filters(makes=(comparison,)) if comparison else None
    if comparison_filters:
        extra = available.filter(pl.col("Make") == comparison)
        data = pl.concat([data, extra]).unique(subset="Make", maintain_order=True).sort("Make")
    if data.is_empty():
        st.info("No selected make has a saved feature row. This does not establish absence of discussion.")
        return
    try:
        inputs, _ = youtube_feature_tables(data, contract["columns"])
    except ValueError as exc:
        st.warning(f"YouTube pricing inputs unavailable: {exc}")
        return
    st.caption("These observations describe makes, not selected years, models, or trims. "
               "Sentiment is experimental; independent automotive accuracy is unmeasured.")
    st.subheader("Discussion topics")
    topics = inputs.filter(pl.col("Group") == "Discussion topics").with_columns(
        pl.col("Feature").str.replace("^yt_", "").str.replace("_mention_share$", "").str.replace_all("_", " ").alias("topic"),
        (pl.col("Value").cast(pl.Float64, strict=False) * 100).alias("share_pct"),
    ).drop_nulls("share_pct")
    if not topics.is_empty():
        chart = alt.Chart(topics.to_pandas()).mark_bar().encode(
            y=alt.Y("topic:N", title=None),
            x=alt.X("share_pct:Q", title="Video-balanced mention share (%)", scale=alt.Scale(domain=[0, 100])),
            color="Make:N", yOffset="Make:N",
            tooltip=["Make:N", alt.Tooltip("topic:N", title="Topic"),
                     alt.Tooltip("share_pct:Q", title="Mention share (%)", format=".1f")],
        ).properties(height=max(370, 24 * topics.height))
        st.altair_chart(chart, alt="Supported make-level discussion topic shares")
    else:
        st.info("No supported topic shares are available for these makes.")
    st.caption("Video-balanced shares of eligible comments. Topics can overlap. "
               "Topic evidence requires 20 eligible comments and five videos.")

    st.subheader("Overall sentiment")
    sentiment = data.filter(~pl.col("yt_overall_missing")).select(
        "Make", "yt_overall_video_balanced_net", "yt_overall_mean_entropy",
        "yt_overall_opinion_count", "yt_overall_opinion_video_count",
    ).drop_nulls("yt_overall_video_balanced_net")
    if not sentiment.is_empty():
        bars = alt.Chart(sentiment.to_pandas()).mark_bar().encode(
            y=alt.Y("Make:N", title=None),
            x=alt.X("yt_overall_video_balanced_net:Q", title="Overall net sentiment (negative to positive)",
                    scale=alt.Scale(domain=[-1, 1])),
            color=alt.condition(alt.datum.yt_overall_video_balanced_net < 0,
                                alt.value("#d97706"), alt.value("#2563eb")),
            tooltip=["Make:N", alt.Tooltip("yt_overall_video_balanced_net:Q", title="Net sentiment", format=".3f"),
                     alt.Tooltip("yt_overall_mean_entropy:Q", title="Tone uncertainty (0–1)", format=".3f"),
                     alt.Tooltip("yt_overall_opinion_count:Q", title="Resolved opinions", format=",.0f"),
                     alt.Tooltip("yt_overall_opinion_video_count:Q", title="Opinion videos", format=",.0f")],
        )
        zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#808080").encode(x="zero:Q")
        st.altair_chart((bars + zero).properties(height=max(140, 32 * sentiment.height)),
                        alt="Overall make sentiment on a fixed scale from minus one to one, centered at zero")
    else:
        st.info("No supported overall sentiment scores are available for these makes.")
    missing_sentiment = sorted(set(data["Make"].to_list()) - set(sentiment["Make"].to_list()))
    if missing_sentiment:
        st.caption("Overall sentiment unavailable: " + "; ".join(missing_sentiment))
    st.caption("Equal-video mean of (positive − negative) / all resolved opinions, including neutral and mixed. "
               "Zero means balanced net sentiment. Missing evidence is omitted. Scores require 20 resolved opinions "
               "and five opinion videos. Hover for support and tone uncertainty; uncertainty is not an error interval.")

    with st.expander("How the YouTube pipeline works", key="youtube_walkthrough", on_change="rerun") as walkthrough:
        if not walkthrough.open:
            return
        st.caption(f"One saved comment for {base_make}. The comparison make applies only to the charts. "
                   "This example is not a representative sample or an explanation of an individual price prediction.")
        comment_args = (contract["directory"], contract["as_of"], contract["source_signature"])
        clauses = query_snapshot(source.identity, youtube_filters, "youtube_trace", *comment_args)
        if clauses.is_empty():
            st.info("No eligible saved comment walkthrough is available for this make.")
        else:
            youtube_comment_trace(clauses)


def youtube_comment_trace(clauses):
    from YOUTUBE.text.features import collapse_polarities

    first = clauses.row(0, named=True)
    st.markdown("**1. Original comment**")
    st.caption(f"{first['Make']} · {first['video_title']} · {first['published_date']} · "
               f"Comment ID: {first['comment_id']}")
    st.text(first["text"])
    st.markdown("**2. Saved clauses, topics, and sentiment**")
    opinions = []
    for index, row in enumerate(clauses.iter_rows(named=True), 1):
        with st.container(border=True):
            st.caption(f"Clause {index} · text positions {row['evidence_start']}–{row['evidence_end']}")
            st.text(row["evidence_text"])
            topics = ", ".join(topic.replace("_", " ") for topic in (row["topics"] or [])) or "None detected"
            st.write(f"Content: {row['content_label'].replace('_', ' ')} · Topics: {topics}")
            st.write(f"Saved overall tone: {row['overall_polarity']}")
            reasons = []
            if not row["feature_eligible"]:
                reasons.append("not eligible vehicle content")
            if row["development_excluded"]:
                reasons.append("development evidence excluded")
            if row["direct_price_mention"]:
                reasons.append("direct numerical price mention")
            st.caption("Feature eligibility: " + ("excluded — " + "; ".join(reasons) if reasons else "eligible"))
            if row["model_truncated"] or row["windowed"]:
                st.caption("Truncated or windowed evidence cannot resolve sentiment.")
            if row["feature_eligible"] and not row["development_excluded"] and row["content_label"] == "vehicle_opinion":
                opinions.append(row["overall_polarity"])
    st.markdown("**3. From comment to make-level features**")
    tone = collapse_polarities(opinions)
    st.write(f"Combined overall opinion: {tone}.")
    st.caption("Saved clause tones combine once per comment: opposing positive and negative tones become mixed; "
               "any unresolved opinion clause leaves the overall opinion unresolved. Facts and questions do not become neutral opinions.")
    st.caption("Eligible topic evidence and resolved opinions feed separate aggregates after date filtering and "
               "within-video text deduplication. Comments aggregate within videos, then videos receive equal weight. "
               "A make-level score requires 20 resolved opinions and five opinion-bearing videos. "
               "Clause eligibility alone does not prove this comment survived deduplication. "
               "The charts above show the final make-level results; no model is rerun here.")



def model_table(data, label, height=None):
    labels = {
        "feature_set": "Feature set", "model": "Model", "price_weighting": "Price weighting",
        "scope": "Evaluation scope", "requested_rows": "Rows matching selected filters", "rows": "Evaluated listings", "test_rows": "Test listings",
        "family_blocks": "Vehicle families", "support": "Sample support", "price_band": "Observed price band (USD)",
        "mae": "MAE (USD)", "rmse": "RMSE (USD)",
        "mse": "MSE (USD²)", "r2": "R²",
        "wape_pct": "WAPE (%)",
        "mean_signed_error": "Bias (USD)",
        "p95_absolute_error": "P95 absolute error (USD)",
        "base_mae": "Base MAE (USD)",
        "enriched_mae": "Enhanced MAE (USD)",
        "enriched_minus_base_mae": "Enhanced − base MAE (USD)",
    }
    columns = {}
    for name in data.columns:
        if name.startswith("validation_"):
            metric = name.removeprefix("validation_")
            columns[name] = "Validation " + labels.get(metric, metric.replace("_", " "))
        else:
            columns[name] = labels.get(name, name.replace("_", " ").capitalize())
    table(data, label, height, column_config=columns)


@st.fragment(run_every="30s")
def watch_price_checkpoints():
    """Refresh the page only when estimator availability or run status changes."""
    try:
        _, _, statuses, _ = diagnostic_inventory()
        identity = statuses.to_json(orient="records")
    except (ValueError, OSError):
        return
    previous = st.session_state.get("price_checkpoint_identity")
    st.session_state["price_checkpoint_identity"] = identity
    if previous is not None and previous != identity:
        st.rerun()
    st.caption("Checks for completed checkpoints every 30 seconds while this tab is open.")


def models(source, filters):
    st.header("Model performance and feature comparisons")
    st.caption("Saved evaluations only · No training, live scoring, or AI calls")
    watch_price_checkpoints()
    if st.button("Refresh saved evaluations", icon=":material/refresh:"):
        st.session_state["artifact_refresh"] = st.session_state.get("artifact_refresh", 0) + 1
    try:
        available, candidates, model_status, inspected = diagnostic_inventory()
    except (ValueError, OSError) as exc:
        st.warning(f"Checkpoint discovery unavailable: {exc}")
        return
    if not model_status.empty:
        completed = model_status.status.eq("completed").sum() if "status" in model_status else len(candidates)
        st.caption(f"{completed} of {len(model_status)} selected estimators have completed evaluation checkpoints.")
        with st.expander("Selected runs and estimator status"):
            model_table(model_status, "Selected training runs, completed estimators and pending checkpoints", 360)
            if not inspected.empty:
                model_table(inspected, "Unavailable checkpoints and their reasons")
    if not available:
        st.info("No completed model evaluations are available.")
        return
    selected = list(available)
    st.caption("Runs may use different data or training protocols; displayed metrics do not establish a matched experiment.")
    st.caption("Only completed estimators from the configured runs appear below. Pending estimators become available as training saves their predictor and both evaluation receipts.")
    mode = st.segmented_control("Evaluation", ["Test", "Validation"], default="Test", key="evaluation_mode")
    strategies = {run.metadata.get("split_strategy") for run in available.values()}
    if strategies == {"cohort_stratified"}:
        st.caption("VINs are disjoint across fit, validation and test within decoded year/make/model groups with at least 25 eligible VINs. The same groups appear in all partitions. Validation evaluates the fit-only model; test evaluates the final refit on fit plus validation.")
    else:
        st.caption("Validation evaluates development fits; test evaluates final refits on separate VINs. Family separation depends on each run's recorded split strategy.")
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
            if "model" in test_report:
                test_report = test_report[test_report.model.isin(allowed_models)]
            if "validation_mae" in test_report:
                metrics = test_report[["model", "validation_mae"]].rename(columns={"validation_mae": "mae"})
                st.caption(f"{run.metadata['model_type']}: recorded selection-validation MAE only; other validation errors are unavailable.")
        if metrics.empty:
            st.info(f"{run.metadata['model_type']}: no saved {mode.lower()} report.")
            continue
        overall = metrics.copy()
        overall.insert(0, "feature_set", run.metadata["model_type"])
        overall["run"] = run.name
        overall["split_strategy"] = run.metadata.get("split_strategy")
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
    st.caption("MAE = mean absolute error; RMSE = root mean squared error; "
               "MSE = mean squared error; WAPE = weighted absolute percentage error; "
               "R² = coefficient of determination; P95 = 95th percentile.")
    leading = [c for c in ["feature_set", "model", "price_weighting", "scope", "mae", "rmse", "r2",
                           "wape_pct", "mean_signed_error", "p95_absolute_error", "test_rows",
                           "validation_mae", "validation_rmse", "validation_r2", "split_strategy"] if c in overall]
    model_table(overall[leading], "Overall saved model performance", 360)
    download(overall, "Download overall metrics", "overall_metrics.csv", "overall_metrics_csv")
    st.caption("Model selection uses recorded validation MAE, not the best exposed test result. Bootstrap MAE intervals describe aggregate error uncertainty, not individual price intervals.")
    if focused:
        filtered = pd.concat(focused, ignore_index=True)
        st.subheader("Performance for dashboard filters")
        st.warning("Narrow filters can match few or no evaluation rows. With fewer than 100 rows, this section broadens to make/model, then make. Each row identifies its actual scope; sidebar filters stay unchanged. Broader scopes drop other vehicle filters.")
        model_table(filtered, "Errors recomputed from matching saved evaluation predictions", 360)
        download(filtered, "Download filtered metrics", "filtered_metrics.csv", "filtered_metrics_csv")
        if "mae" in filtered:
            plotted = filtered.dropna(subset=["mae"]).copy()
            if not plotted.empty:
                chart = alt.Chart(plotted).mark_bar().encode(
                    x=alt.X("mae:Q", title=f"Filtered {mode.lower()} MAE (USD)"),
                    y=alt.Y("model:N", title="Model", sort=candidates.model.drop_duplicates().tolist()),
                    yOffset=alt.YOffset("feature_set:N", sort=candidates.feature_set.drop_duplicates().tolist()),
                    color=alt.Color("feature_set:N", title="Feature set", sort=candidates.feature_set.drop_duplicates().tolist()),
                    tooltip=["feature_set", "model", "scope", "rows", "mae", "support"])
                st.altair_chart(chart, alt="Filtered held-out MAE by feature set and estimator")

        st.caption("Low support means fewer than 100 evaluation rows or five vehicle families. Empty selections and undefined R² stay unavailable.")
    st.subheader("Prediction diagnostics and feature importance")
    st.caption("Historical diagnostics use saved evaluation prices and vehicle attributes. Filters requiring attributes absent from the saved receipt also require an unchanged source snapshot.")
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
        st.info("No explanation has been cached for this checkpoint. Run python -m car_app.model_registry --refresh-diagnostic-explanations to build it from the saved model and frozen inputs.")
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
