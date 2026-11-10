"""Short questionnaire and AI-selected historical vehicle recommendations."""
import json
import pandas as pd
import polars as pl
import streamlit as st

from car_app.data import COHORT, ROOT, Filters, filter_options, query_snapshot, snapshot
from car_app.requirements import (EQUIPMENT, PRIORITIES, USE_CASES, QuestionDraft,
                                  amount_label, budget_amount, mileage_amount, range_presets)
from car_app.recommendation import load_similarity, category_explorer, SIMILARITY_LABELS
from car_app.model_registry import digest, price_estimates, registry
from car_app.artifacts import stat_key
from car_app.state import update_result_context
from car_app.discovery import active_catalog as active_discovery, COMPONENTS as ISSUE_COMPONENTS, profile_names, discovery_caption
from car_app.agent_ui import render_chat, submit_request, sync_question_widgets


def show_table(frame, label):
    st.dataframe(frame, hide_index=True, width="stretch", alt=label)


source = snapshot()
filters = st.session_state["dashboard_filters"]
session = st.session_state["recommendation_session"]
st.caption("Historical listings and advertised USD prices through 2020. Mileage is shown in source units. Current availability is unknown.")
config, catalog, discovery = {}, None, None
artifact_warnings = []
try:
    config = registry(include_price_diagnostics=False)
except (OSError, ValueError, KeyError) as exc:
    artifact_warnings.append(f"Saved artifact registry unavailable: {exc}")
try:
    discovery = active_discovery(source)
except (OSError, ValueError, KeyError) as exc:
    artifact_warnings.append(f"Saved discovery bundle unavailable: {exc}")
if discovery is None and not config.get("discovery_run"):
    try:
        run = config["similarity_run"]
        if config.get("similarity_manifest_sha256") and digest(ROOT / "MODEL_OUTPUT" / run / "manifest.json") != config["similarity_manifest_sha256"]:
            raise ValueError("Pinned similarity receipt changed")
        catalog = load_similarity(run, stat_key(ROOT / "MODEL_OUTPUT" / run / "manifest.json"), str(source.directory), source.identity)
    except (OSError, ValueError, KeyError) as exc:
        artifact_warnings.append(f"Saved similarity artifacts unavailable: {exc}")
cohorts, observed = filter_options(source.identity)
categories = {"makes": sorted(cohorts["Make"].drop_nulls().unique().to_list()),
    "body_types": sorted(observed["body_type"]), "fuels": sorted(observed["fuel_type"]),
    "drivetrains": sorted(observed["wheel_system"]), "transmissions": sorted(observed["transmission"]),
    "vehicle_categories": catalog.category_choices() if catalog is not None else {}}
category_identity = str(config.get("discovery_manifest_sha256") or config.get("similarity_manifest_sha256", config.get("similarity_run"))) + (str(catalog.path) if catalog else "")
if st.session_state.get("category_widget_identity") != category_identity:
    for widget in list(st.session_state):
        if widget.startswith("rec_category_") or widget.startswith("rec_categories_"):
            del st.session_state[widget]
    if session.requirements.get("category_preferences"):
        session.requirements = {}
        session.agent_result = None
        session.messages = []
    st.session_state["category_widget_identity"] = category_identity
sync_question_widgets(session)
ranges = query_snapshot(source.identity, Filters(), "summary")
budget_values = range_presets(ranges["min_price"], ranges["max_price"])
budget_options = [amount_label(value, "$") for value in budget_values]
mileage_values = range_presets(ranges["min_mileage"], ranges["max_mileage"])
mileage_options = [amount_label(value) for value in mileage_values]
for key, options in (("rec_budget", budget_options), ("rec_mileage", mileage_options)):
    current = st.session_state.get(key)
    if current is not None and current not in options:
        options.append(current)
with st.container(border=True):
    st.header("Find your vehicle")
    st.caption("Answer six short questions. The AI will choose a best match and up to five alternatives. These answers are independent of the Dashboard filters.")
    with st.form("recommendation_questions"):
        left, right = st.columns(2)
        with left:
            budget_label = st.selectbox("1. What is your maximum budget (USD)?", budget_options,
                index=None if "rec_budget" in st.session_state or not budget_values else
                    budget_values.index(min(budget_values, key=lambda value: abs(value - 25000))),
                accept_new_options=True, placeholder="Choose or type a budget",
                help="Choose a preset or type a custom amount, such as $18,000 or 18k, and press Enter.", key="rec_budget")
            if budget_values:
                st.caption(f"Observed price range: {amount_label(ranges['min_price'], '$')}–{amount_label(ranges['max_price'], '$')}.")
            bodies = st.multiselect("2. What vehicle types would you consider?", categories["body_types"],
                placeholder="No preference", key="rec_bodies")
            mileage_label = st.selectbox("3. What is your maximum mileage (source units)?", mileage_options,
                index=None, accept_new_options=True, placeholder="No preference; choose or type mileage",
                help="Choose a preset or type a custom amount, such as 75,000 or 75k, and press Enter. Clear to remove the limit.", key="rec_mileage")
            if mileage_values:
                st.caption(f"Observed mileage range: {amount_label(ranges['min_mileage'])}–{amount_label(ranges['max_mileage'])} source units.")
        with right:
            seats = st.number_input("4. How many seats do you need?", min_value=1, value=None,
                step=1, placeholder="No preference", key="rec_seats")
            use_case = st.selectbox("5. What will you mainly use it for?", USE_CASES,
                index=None if "rec_use_case" in st.session_state else len(USE_CASES) - 1, key="rec_use_case")
            priorities = st.multiselect("6. What matters most?", PRIORITIES, max_selections=2,
                help="Select your primary priority first. A second priority is optional.", key="rec_priorities")
            st.caption("Select in priority order. Unsupported needs remain explicit evidence gaps.")
        category_preferences, category_mode = {}, "prefer"
        evidence_period, reference_profile_id, avoided_components = "historical", None, []
        with st.expander("Additional preferences"):
            st.caption("Optional limits. Unknown facts cannot satisfy a must-have. Seller reports do not verify vehicle condition.")
            condition = st.selectbox("New, used, or either?", ["Not decided", "Either", "Used", "New"],
                index=None if "rec_condition" in st.session_state else 0, key="rec_condition")
            years = st.columns(2)
            with years[0]:
                min_year = st.number_input("First model year", min_value=1900, value=None, step=1,
                    placeholder="Any year", key="rec_min_year")
            with years[1]:
                max_year = st.number_input("Last model year", min_value=1900, value=None, step=1,
                    placeholder="Any year", key="rec_max_year")
            makes = st.multiselect("Acceptable makes", categories["makes"], key="rec_makes")
            fuels = st.multiselect("Acceptable fuel types", categories["fuels"], key="rec_fuels")
            drives = st.multiselect("Acceptable drivetrains", categories["drivetrains"], key="rec_drives")
            transmissions = st.multiselect("Acceptable transmissions", categories["transmissions"], key="rec_transmissions")
            horsepower = st.number_input("Minimum advertised horsepower", min_value=0., value=None,
                step=25., placeholder="No preference", key="rec_hp")
            equipment = st.multiselect("Required advertised equipment", EQUIPMENT, key="rec_equipment")
            history = st.checkbox("Require explicit reports of no accidents, frame damage, salvage, or theft title", key="rec_history")
            for dimension, values in categories["vehicle_categories"].items():
                category_preferences[dimension] = tuple(st.multiselect(dimension.replace("_", " ").capitalize(),
                    values, key="rec_category_" + dimension))
            if categories["vehicle_categories"]:
                category_mode = st.selectbox("Category matching", ["prefer", "strict"], key="rec_categories_mode")
            if discovery is not None:
                reference_names = profile_names(discovery.profiles)
                reference_profile_id = st.selectbox("Reference vehicle for optional specification and crash comparisons",
                    [None, *reference_names], format_func=lambda value: "No reference" if value is None else reference_names[value],
                    key="rec_discovery_reference")
                evidence_period = st.selectbox("Reported-issue evidence", discovery.receipt["periods"],
                    format_func=lambda value: "Historical through 2020" if value == "historical" else "Latest saved evidence; historical prices",
                    key="rec_evidence_period")
                avoided_components = st.multiselect("Reported component concerns to consider", ISSUE_COMPONENTS, key="rec_avoid_components")
            st.caption("Price evidence: best completed tree and linear checkpoints by validation MAE. "
                       "Historical value ranking uses saved held-out test predictions; other estimates are supplemental.")
        submitted = st.form_submit_button("Get AI recommendations", key="recommendation_search", type="primary",
            icon=":material/auto_awesome:", disabled=session.request_id is not None)
    st.caption("Submitting these answers or a follow-up sends the necessary vehicle context to OpenAI. Ordinary page interactions make no AI request.")

identity = source.identity + json.dumps(config, sort_keys=True) + (str(catalog.path) if catalog else "")
if discovery is not None:
    identity += str(discovery.path) + discovery.receipt["completed_at"]
update_result_context(session, identity)
update_result_context(st.session_state["similarity_session"], identity)
if submitted:
    try:
        if not priorities:
            raise ValueError("Choose your main priority so the AI can explain which vehicle fits best.")
        draft = QuestionDraft(use_case=use_case, max_price_usd=budget_amount(budget_label), condition=condition,
            body_types=tuple(bodies), min_seats=int(seats) if seats is not None else None,
            max_mileage=mileage_amount(mileage_label),
            min_year=int(min_year) if min_year is not None else None, max_year=int(max_year) if max_year is not None else None,
            makes=tuple(makes), fuels=tuple(fuels), drivetrains=tuple(drives), transmissions=tuple(transmissions),
            min_horsepower=float(horsepower) if horsepower is not None else None, required_equipment=tuple(equipment),
            require_reported_clean_history=history, priorities=tuple(priorities), category_preferences=category_preferences,
            category_mode=category_mode, evidence_period=evidence_period, reference_profile_id=reference_profile_id,
            avoided_components=tuple(avoided_components))
        draft.validate(categories)
        session.revision += 1
        session.requirements = draft.contract(source.identity, session.revision)
        session.messages, session.result, session.agent_result = [], None, None
        st.session_state["rec_form_revision"] = session.revision
        submit_request(session, source, config, categories, discovery, catalog,
            message="Use my submitted answers and ordered priorities to choose one best match and up to five distinct alternatives. Explain why each fits, its main tradeoff, and how it compares with the best match. Optional unanswered questions do not block this request.")
    except (TypeError, ValueError) as exc:
        st.error(str(exc))
if session.requirements:
    if session.requirements.get("snapshot_identity") != source.identity:
        st.warning("The saved source changed. Submit your answers again before requesting recommendations.")
    else:
        render_chat(session, source, config, categories, discovery, catalog, key="recommendation_chat")
with st.expander("Explore vehicle groups and research"):
    for warning in artifact_warnings:
        st.caption(warning)
    if st.checkbox("Load the saved vehicle explorer", key="rec_show_research"):
        if discovery is not None:
            discovery_caption(discovery)
            show_table(pd.DataFrame(discovery.validation["checks"]), "Saved discovery validation checks")
            from car_app.discovery import peer_review_panel
            peer_review_panel(discovery)
        if discovery is not None:
            from car_app.discovery import similarity_dashboard
            with st.container(border=True):
                try:
                    similarity_dashboard(source, filters)
                    reference = st.session_state.get("v6_similarity_reference")
                    grain = st.session_state.get("v6_similarity_grain") or "model"
                    if reference:
                        from car_app.agent_ui import render_chat, sync_reference
                        similar_session = st.session_state["similarity_session"]
                        sync_reference(similar_session, source.identity + filters.identity() + grain + reference)
                        render_chat(similar_session, source, config, categories, active_discovery(source, grain),
                                    key="similarity_chat", reference_id=reference, similarity=True)
                except (OSError, ValueError, KeyError) as exc:
                    st.warning(f"Saved v6 similarity unavailable: {exc}")
        else:
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
