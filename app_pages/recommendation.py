"""Historical requirements search, frozen similarity and explanatory clusters."""
import json
import pandas as pd
import polars as pl
import streamlit as st

from car_app.data import COHORT, ROOT, filter_options, query_snapshot, snapshot
from car_app.requirements import EQUIPMENT, PRIORITIES, USE_CASES, QuestionDraft
from car_app.recommendation import SORTS, load_similarity, search_candidates, category_explorer, SIMILARITY_LABELS, select_discovery_grain
from car_app.model_registry import digest, price_estimates, registry
from car_app.artifacts import stat_key, diagnostic_inventory
from car_app.state import update_result_context


def show_table(frame, label):
    st.dataframe(frame, hide_index=True, width="stretch", alt=label)


def with_listing_clusters(frame, catalog):
    if frame.is_empty() or catalog is None:
        return frame
    if "profile_id" in frame.columns:
        frame = frame.drop("profile_id")
    membership = pl.scan_parquet(catalog.membership_path()).select("vin", "profile_id").filter(
        pl.col("vin").is_in(frame["vin"].to_list())).collect()
    labels = []
    for profile in membership["profile_id"].unique():
        row = catalog.cluster_rows([profile], catalog.context(profile)).iloc[0]
        labels.append({"profile_id": profile, "cluster": row.cluster, "cluster_description": row.description,
                       "cluster_status": row.cluster_status, "cluster_context": row.cluster_context,
                       **({"category_summary": row.category_summary} if "category_summary" in row and "category_summary" not in frame.columns else {})})
    if not labels:
        return frame
    return frame.join(membership, on="vin", how="left", validate="1:1").join(
        pl.from_pandas(pd.DataFrame(labels)), on="profile_id", how="left", validate="m:1")


def show_search(result, config, catalog):
    st.caption(f"{result['eligible_listings']:,} eligible historical listings. Sort: {result['sort_by']}. "
               "Priorities are context; no personalized utility score is inferred.")
    show_table(result["cohorts"], "Up to five cohorts summarized from qualifying members only")
    listings = result["listings"]
    displayed = [*COHORT, "category_summary", "category_match", "price", "predicted_price_usd", "mileage", "cluster_description", "cluster_status",
                 "body_type", "seats", "horsepower", "vin", "listing_id", "price_model", "cluster_context"]
    show_table(listings.select([c for c in displayed if c in listings.columns]),
               "Up to ten qualifying listings with separate price estimates and cluster contexts")
    if result.get("category_preferences"):
        st.caption("Category mode: " + result["category_mode"] + ". Match fraction counts requested category dimensions; it is not a confidence or suitability score. Assignments are provisional.")
    st.caption("Cohort quartiles describe qualifying members. Cluster numbers are comparable only within the stated fitted model. "
               "Predicted prices do not affect eligibility or ordering; this is not current inventory.")
    if result.get("price_warning"):
        st.warning(result["price_warning"])
    with st.expander("Applied requirements and exclusions"):
        show_table(result["exclusions"], "Sequential exclusions from the independent requirements")
    st.download_button("Download returned listings", listings.write_csv(), "recommended_listings.csv", key="rec_results_csv")


source = snapshot()
filters = st.session_state["dashboard_filters"]
session = st.session_state["recommendation_session"]

st.info("Historical-data prototype. The active saved run supplies categories, cluster diagrams and similarity alternatives. No AI calls run.")
config, catalog = {}, None
try:
    config = registry()
    run = config["similarity_run"]
    if config.get("similarity_manifest_sha256") and digest(ROOT / "MODEL_OUTPUT" / run / "manifest.json") != config["similarity_manifest_sha256"]:
        raise ValueError("Pinned similarity receipt changed")
    catalog = load_similarity(run, stat_key(ROOT / "MODEL_OUTPUT" / run / "manifest.json"), str(source.directory), source.identity, select_discovery_grain(run, "recommendation_discovery_grain"))
except (OSError, ValueError, KeyError) as exc:
    st.warning(f"Saved similarity artifacts unavailable: {exc}")
with st.expander("Price model and provenance"):
    try:
        _, choices, _, _ = diagnostic_inventory()
        if not choices.empty:
            options = choices.label.tolist()
            active = config.get("active_price", {})
            default = active.get("run", "base/").split("/")[0] + "-" + active.get("model", "catboost")
            if st.session_state.get("price_model_choice") not in options:
                st.session_state["price_model_choice"] = default if default in options else options[0]
            selected = st.selectbox("Price estimate model", options, key="price_model_choice")
            choice = choices.loc[choices.label.eq(selected)].iloc[0]
            path = ROOT / "MODEL_OUTPUT" / choice.run
            config = {**config, "active_price": {"run": choice.run, "model": choice.model,
                "metadata_sha256": digest(path / "run_metadata.json"),
                "predictor_sha256": digest(path / choice.model / "predictor.joblib")}}
            st.caption("Each feature-set/model combination uses its newest completed full-data checkpoint. Choices refresh on each app interaction.")
        else:
            config = {**config, "active_price": None}
            st.info("No completed price models available; observed-price search remains available.")
    except (OSError, ValueError, KeyError) as exc:
        config = {**config, "active_price": None}
        st.warning(f"Model discovery unavailable: {exc}")
identity = source.identity + json.dumps(config, sort_keys=True) + (str(catalog.path) if catalog else "")
update_result_context(session, identity)
update_result_context(st.session_state["similarity_session"], identity)
category_identity = str(config.get("similarity_manifest_sha256", config.get("similarity_run"))) + (str(catalog.path) if catalog else "")
if st.session_state.get("category_widget_identity") != category_identity:
    for widget in list(st.session_state):
        if widget.startswith("rec_category_") or widget.startswith("rec_categories_"):
            del st.session_state[widget]
    if session.requirements.get("category_preferences"):
        session.requirements = {}
    st.session_state["category_widget_identity"] = category_identity
with st.container(border=True):
    st.header("Vehicle recommender")
    st.caption("Independent requirements · Dashboard filters do not restrict this draft")
    st.markdown("**1. Purpose and budget → 2. Requirements → 3. Priorities → 4. Review and search**")
    st.caption("Use historical advertised USD prices. Mileage is shown in source units until verified.")
    cohorts, observed = filter_options(source.identity)
    categories = {
        "makes": sorted(cohorts["Make"].drop_nulls().unique().to_list()),
        "body_types": sorted(observed["body_type"]),
        "fuels": sorted(observed["fuel_type"]),
        "drivetrains": sorted(observed["wheel_system"]),
        "transmissions": sorted(observed["transmission"]),
        "vehicle_categories": catalog.category_choices() if catalog is not None else {},
    }
    with st.form("recommendation_questions"):
        use_case = st.selectbox("What will you use the vehicle for most often?", USE_CASES,
                                index=len(USE_CASES) - 1, key="rec_use_case")
        budget_known = st.checkbox("I have a maximum advertised-price budget", key="rec_budget_known")
        budget = st.number_input("Maximum advertised price (USD)", min_value=1, value=25000,
                                 step=1000, key="rec_budget")
        condition = st.selectbox("New, used, or either?", ["Not decided", "Either", "Used", "New"],
                                 key="rec_condition")
        st.subheader("Must-have requirements")
        st.caption("Leave optional selections empty for no restriction. Unknown facts cannot satisfy a must-have.")
        bodies = st.multiselect("Acceptable body styles", categories["body_types"], key="rec_bodies")
        seats_known = st.checkbox("I need a minimum number of seats", key="rec_seats_known")
        seats = st.number_input("Minimum advertised seats", min_value=1, value=5, step=1, key="rec_seats")
        mileage_known = st.checkbox("I have a maximum advertised-mileage limit", key="rec_mileage_known")
        mileage = st.number_input("Maximum advertised mileage (source units)", min_value=0,
                                  value=100000, step=10000, key="rec_mileage")
        years_known = st.checkbox("I want a specific decoded model-year range", key="rec_years_known")
        min_year = st.number_input("First model year", min_value=1900, value=2010, step=1, key="rec_min_year")
        max_year = st.number_input("Last model year", min_value=1900, value=2020, step=1, key="rec_max_year")
        makes = st.multiselect("Acceptable makes", categories["makes"], key="rec_makes")
        fuels = st.multiselect("Acceptable fuel types", categories["fuels"], key="rec_fuels")
        drives = st.multiselect("Acceptable drivetrains", categories["drivetrains"], key="rec_drives")
        history = st.checkbox("Require explicit reports of no accidents, frame damage, salvage, or theft title",
                              key="rec_history")
        st.caption("This filter excludes unknown history. Seller reports are not independent verification.")
        st.subheader("Equipment and enthusiast needs (optional)")
        transmissions = st.multiselect("Acceptable advertised transmissions", categories["transmissions"],
                                        key="rec_transmissions")
        horsepower_known = st.checkbox("I need a minimum advertised horsepower", key="rec_hp_known")
        horsepower = st.number_input("Minimum advertised horsepower", min_value=0, value=200,
                                     step=25, key="rec_hp")
        equipment = st.multiselect("Required advertised equipment", EQUIPMENT, key="rec_equipment")
        st.caption("Equipment must be advertised on the listing. Horsepower alone does not establish acceleration, handling, or towing.")
        category_preferences = {}
        category_mode = "prefer"
        if categories["vehicle_categories"]:
            st.subheader("Vehicle categories (optional)")
            st.caption("Source-supported labels are provisional. Prefer matches, or explicitly require matching categories. Unknowns do not satisfy strict matching.")
            for dimension, values in categories["vehicle_categories"].items():
                category_preferences[dimension] = tuple(st.multiselect(
                    dimension.replace("_", " ").capitalize(), values,
                    format_func=lambda value: value.replace("_", " ").capitalize(), key="rec_category_" + dimension))
            category_mode = st.selectbox("Category matching", ["prefer", "strict"],
                format_func=lambda value: "Prefer matching categories" if value == "prefer" else "Only matching categories", key="rec_categories_mode")
        st.subheader("What matters most?")
        priorities = st.multiselect("Choose up to three priorities to discuss", PRIORITIES,
                                    max_selections=3, key="rec_priorities")
        st.caption("Priorities guide follow-up questions. Ranking weights have not been selected.")
        submitted = st.form_submit_button("Review my requirements", key="rec_review",
                                          icon=":material/fact_check:")
    if submitted:
        try:
            draft = QuestionDraft(
                use_case=use_case, max_price_usd=float(budget) if budget_known else None,
                condition=condition, body_types=tuple(bodies), min_seats=int(seats) if seats_known else None,
                max_mileage=float(mileage) if mileage_known else None,
                min_year=int(min_year) if years_known else None, max_year=int(max_year) if years_known else None,
                makes=tuple(makes), fuels=tuple(fuels), drivetrains=tuple(drives),
                transmissions=tuple(transmissions), min_horsepower=float(horsepower) if horsepower_known else None,
                required_equipment=tuple(equipment),
                require_reported_clean_history=history, priorities=tuple(priorities),
                category_preferences={d: values for d, values in category_preferences.items() if values}, category_mode=category_mode,
            )
            draft.validate(categories)
            session.revision += 1
            session.requirements = draft.contract(source.identity, session.revision)
            session.result = None
        except (TypeError, ValueError) as exc:
            st.error(str(exc))
    if session.requirements:
        st.subheader("Submitted question draft")
        st.caption("This reflects your last review submission; submit again to apply edits.")
        if session.requirements["snapshot_identity"] != source.identity:
            st.warning("The snapshot changed. Review and submit this draft again before using it.")
        st.markdown("**Primary use:** " + session.requirements["use_case"])
        st.json(session.requirements["hard_requirements"])
        st.markdown("**Priorities:** " + (", ".join(session.requirements["soft_priorities"]) or "Not selected"))
        if session.requirements.get("category_preferences"):
            st.write("Vehicle categories (" + session.requirements["category_mode"] + ")", session.requirements["category_preferences"])
        pending = session.requirements["unresolved_questions"]
        if pending:
            st.markdown("**Unresolved context (not converted into scores or restrictions):**")
            for question in pending:
                st.markdown("- " + question)
        else:
            st.success("The first-round questions are answered. Search uses your explicit requirements.")
        st.download_button("Download question draft", json.dumps(session.requirements, indent=2),
                           file_name="vehicle_requirements.json", mime="application/json",
                           key="rec_download")
    sort_by = st.selectbox("Order eligible listings", list(SORTS), key="rec_sort")
    if session.result and session.result.get("sort_by") != sort_by:
        session.result = None
    ready = bool(session.requirements and session.requirements.get("snapshot_identity") == source.identity
                 and session.requirements.get("hard_requirements", {}).get("max_price_usd") is not None)
    if st.button("Generate recommendations", disabled=not ready, key="recommendation_search"):
        try:
            with st.spinner("Applying requirements and reading the selected price model…"):
                result = search_candidates(source, session.requirements, session.revision, sort_by, catalog=catalog)
                enriched, warning = price_estimates(result["listings"], config)
                result["listings"] = with_listing_clusters(enriched, catalog)
                result["price_warning"] = warning
                session.result = result
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            st.error(str(exc))
    if session.result:
        show_search(session.result, config, catalog)
    st.caption("Broad priorities remain unscored. Only the explicit requirements above filter results.")

scoped_profiles = None
if catalog is not None:
    with st.spinner("Matching the dashboard filters to saved vehicle profiles…"):
        scoped_profiles = catalog.filtered_profiles(source, filters)

with st.container(border=True):
    st.header("Similar vehicles to the filter in the dashboard")
    st.caption("Reference scope follows the dashboard; alternatives will search the broader catalog.")
    active = filters.active()
    if st.session_state.get("filter_makes") is None:
        active.pop("makes", None)
    if not active:
        st.info("Choose a reference below, or narrow it with Dashboard filters. Alternatives use the broader catalog.")
    else:
        labels = {"makes": "Make", "models": "Model", "years": "Model year", "trims": "Trim",
                  "body_types": "Body style", "fuels": "Fuel", "drivetrains": "Drivetrain",
                  "condition": "Condition", "min_price": "Minimum price (USD)",
                  "max_price": "Maximum price (USD)", "max_mileage": "Maximum advertised mileage"}
        for field, value in active.items():
            display = ", ".join(map(str, value)) if isinstance(value, tuple) else str(value)
            st.markdown(f"**{labels[field]}:** {display}")
        with st.spinner("Reading the dashboard reference…"):
            summary = query_snapshot(source.identity, filters, "summary")
        with st.container(horizontal=True):
            st.metric("Reference listings", f"{summary['listings']:,}", border=True)
            if catalog is not None and catalog.manifest.get("grain")=="model":
                st.metric("Model-level reference profiles",f"{int(scoped_profiles.identity_complete.sum()):,}",border=True)
            else:
                st.metric("Complete reference cohorts", f"{summary['cohorts']:,}", border=True)
            price = summary["median_price"]
            st.metric("Reference median advertised price", f"$ {price:,.0f}" if price is not None else "Unavailable",
                      border=True)
        if not summary["listings"]:
            st.warning("These dashboard filters match no listings.")
    similar_session = st.session_state["similarity_session"]
    profile = None
    approach = "direct_specs"
    if catalog is not None:
        choices = scoped_profiles if catalog.manifest.get("grain") == "model" else scoped_profiles[scoped_profiles.complete_matching_listings.gt(0)]
        if "identity_complete" in choices:
            choices = choices[choices.identity_complete]
        scope = filters.identity() + category_identity
        if st.session_state.get("similarity_widget_scope") != scope:
            st.session_state.pop("similarity_profile", None)
            st.session_state["similarity_widget_scope"] = scope
        st.caption("All matching VIN-supported variants are available. Body, fuel, drivetrain, condition, price and mileage filters apply to reference membership. Results update automatically.")
        approaches = catalog.approaches()
        if st.session_state.get("similarity_approach") not in approaches:
            st.session_state.pop("similarity_approach", None)
        approach = st.selectbox("Similarity approach", approaches, format_func=SIMILARITY_LABELS.get, key="similarity_approach")
        if not choices.empty:
            names = {row.profile_id: " / ".join(str(getattr(row, c)) for c in COHORT if pd.notna(getattr(row,c)))
                     + f" / {row.body_style} / {row.powertrain} / {int(row.matching_listings):,} matching listings" for row in choices.itertuples()}
            if st.session_state.get("similarity_profile") not in names:
                st.session_state.pop("similarity_profile", None)
            profile = st.selectbox("Reference cohort and variant", list(names), index=0 if active else None,
                                   format_func=names.get, key="similarity_profile")
        else:
            st.info("No complete VIN-supported reference profiles match these filters.")
    reference_requirements = {"profile_id": profile, "approach": approach, "filter_identity": filters.identity()}
    if similar_session.requirements != reference_requirements:
        similar_session.requirements = reference_requirements
        similar_session.revision += 1
        similar_session.result = None
    refresh = st.button("Find similar vehicles", disabled=profile is None, key="similarity_search")
    if refresh or (profile is not None and similar_session.result is None):
        try:
            with st.spinner("Reading saved similarity alternatives and cluster evidence…"):
                result = catalog.similar(profile, source.identity, similar_session.revision, approach=approach)
                ids = [profile, *result["alternatives"].peer_profile_id.tolist()]
                market, examples = catalog.market_rows(source, ids, reference_id=profile, reference_filters=filters)
                priced, warning = price_estimates(examples, config)
                result["market"] = market
                result["examples"] = priced
                result["price_warning"] = warning
                similar_session.result = result
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            st.error(str(exc))
    if similar_session.result:
        result = similar_session.result
        reference_columns = [*COHORT, "category_summary", "body_style", "powertrain", "description", "cluster_status"]
        show_table(result["reference"][[c for c in reference_columns if c in result["reference"]]],
                   "Reference variant and cluster membership")
        st.caption("Shared fitted context: " + " / ".join(map(str, result["cluster_context"]))
                   + ". Unsupported membership does not mean a different cluster.")
        alternatives = result["alternatives"]
        columns = ["rank", *COHORT, "category_summary", "description", "shared_cluster", "cluster_status", "category_match", "cluster_penalty", "distance", "ranking_distance", "year_window",
                   "body_style", "powertrain", "shared_size", "shared_powertrain"]
        show_table(alternatives[[c for c in columns if c in alternatives]], "Saved similarity alternatives with category and cluster evidence")
        ranking_note = {"category_cluster": "Reference category matches rank first; supported cluster mismatches add a soft distance penalty. Category and cluster usefulness remain provisional.",
            "category_specs": "Reference category matches rank first; specification distance breaks ties.",
            "direct_specs": "Specification distance determines ranking."}[result["approach"]]
        st.caption(f"Returned {len(alternatives)} of five alternatives. Missing specifications and unknown electrification remain explicit. "
                   + ranking_note + " Match fractions are not confidence scores; distance is not a buyer-suitability score.")
        show_table(result["market"], "Observed price distributions for the reference and returned VIN-supported variants")
        example_columns = [*COHORT, "price", "predicted_price_usd", "mileage", "profile_id", "vin", "price_model"]
        show_table(result["examples"].select([c for c in example_columns if c in result["examples"].columns]),
                   "Example listings: lowest observed price per variant, with individual model estimates")
        st.caption("Reference prices and example listings follow the dashboard filters. Alternative prices describe the broader catalog. Predictions apply to individual example listings.")
        if result.get("price_warning"):
            st.warning(result["price_warning"])
        with st.expander("Specification comparisons and query status"):
            show_table(result["evidence"], "Shared and missing comparison fields for each alternative")
            st.json(result["status"])
    st.caption("Reference changes clear similarity results while preserving your independent question draft.")
if catalog is not None:
    category_explorer(catalog, "recommendation_categories", profiles=scoped_profiles, reference_id=profile)
