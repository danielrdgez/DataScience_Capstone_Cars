"""Native Streamlit entry points for the same bounded recommendation service."""
import json
from uuid import uuid4

import pandas as pd
import streamlit as st

from car_app.agent import markdown_receipt, run_turn, RESULT_SCHEMA_VERSION
from car_app.agent_context import identity, dictionary_context
from car_app.data import COHORT
from car_app.requirements import amount_label


def sync_reference(panel, reference):
    if panel.chat_reference != reference:
        panel.chat_reference = reference
        panel.chat_requirements = {}
        panel.messages = []
        panel.agent_result = None
        panel.request_id = None
        panel.revision += 1


def sync_question_widgets(panel):
    """Reflect successful chat edits in the form before its widgets are created."""
    if st.session_state.get("rec_form_version") != 2:
        for flag, field in (("rec_mileage_known", "rec_mileage"), ("rec_seats_known", "rec_seats"),
                            ("rec_hp_known", "rec_hp"), ("rec_years_known", "rec_min_year"),
                            ("rec_years_known", "rec_max_year")):
            if flag in st.session_state and not st.session_state[flag]:
                st.session_state[field] = None
        old_budget = st.session_state.get("rec_budget")
        if isinstance(old_budget, (int, float)):
            st.session_state["rec_budget"] = f"${old_budget:,}"
        if "rec_priorities" in st.session_state:
            st.session_state["rec_priorities"] = st.session_state["rec_priorities"][:2]
        st.session_state["rec_form_version"] = 2
    if isinstance(st.session_state.get("rec_mileage"), (int, float)):
        st.session_state["rec_mileage"] = amount_label(st.session_state["rec_mileage"])
    if not panel.requirements or st.session_state.get("rec_form_revision") == panel.revision:
        return
    from car_app.agent import preference_values
    values = preference_values(panel.requirements)
    budget = values["max_price_usd"]
    mapping = {"rec_use_case": values["use_case"],
        "rec_budget": amount_label(budget, "$"),
        "rec_condition": values["condition"], "rec_bodies": list(values["body_types"]),
        "rec_seats": values["min_seats"], "rec_mileage": amount_label(values["max_mileage"]),
        "rec_min_year": values["min_year"], "rec_max_year": values["max_year"],
        "rec_makes": list(values["makes"]), "rec_fuels": list(values["fuels"]),
        "rec_drives": list(values["drivetrains"]), "rec_transmissions": list(values["transmissions"]),
        "rec_hp": values["min_horsepower"], "rec_equipment": list(values["required_equipment"]),
        "rec_history": values["require_reported_clean_history"], "rec_priorities": list(values["priorities"])[:2],
        "rec_evidence_period": values["evidence_period"], "rec_discovery_reference": values["reference_profile_id"],
        "rec_avoid_components": list(values["avoided_components"])}
    st.session_state.update(mapping)
    st.session_state["rec_form_revision"] = panel.revision


def submit_request(panel, source, config, categories, catalog=None, legacy=None, *, message,
                   reference_id=None, similarity=False, regenerate=False):
    request, revision, artifact = str(uuid4()), panel.revision, panel.artifact_identity
    panel.request_id = request
    history = panel.messages[:-2] if regenerate else panel.messages
    current = panel.chat_requirements if similarity else panel.requirements
    try:
        with st.spinner("The AI is choosing vehicles and checking the supporting facts..."):
            result = run_turn(source, current, revision, categories, message, history, config, catalog, legacy, reference_id)
        if panel.request_id != request or panel.revision != revision or panel.artifact_identity != artifact:
            st.info("The requirements or saved models changed. Submit again for current recommendations.")
            return False
        panel.revision = result["revision"]
        if similarity:
            panel.chat_requirements = result["requirements"]
        else:
            panel.requirements = result["requirements"]
            panel.result = None
        panel.agent_result = result
        panel.messages = [*history, {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"]["message"]}][-20:]
        return True
    except (ValueError, OSError) as exc:
        st.warning(str(exc))
        return False
    finally:
        if panel.request_id == request:
            panel.request_id = None


def vehicle_title(candidate):
    return " ".join(str(candidate[c]) for c in COHORT if candidate.get(c) is not None)


def render_vehicle(item, candidate, rank):
    with st.container(border=True):
        st.subheader(("Best AI match: " if rank == 0 else f"{rank}. ") + vehicle_title(candidate))
        price, mileage, seats = candidate.get("price"), candidate.get("mileage"), candidate.get("seats")
        facts = [f"${price:,.0f} advertised" if price is not None else "Price unavailable",
                 f"{mileage:,.0f} mileage (source units)" if mileage is not None else "Mileage unavailable",
                 str(candidate.get("body_type") or "Body type unavailable"),
                 f"{seats:g} seats" if seats is not None else "Seating unavailable"]
        st.caption(" · ".join(facts))
        st.markdown("**Why this fits you:** " + item["reason"])
        st.markdown("**Main tradeoff:** " + item["tradeoff"])
        st.markdown(("**Why it leads:** " if rank == 0 else "**Choose this instead when:** ") + item["comparison"])


def render_result(result, key, panel=None, source=None, config=None, legacy=None):
    failure = result["telemetry"]["failure_category"]
    if any("price checkpoint unavailable" in warning.lower() for warning in result.get("warnings", [])):
        st.warning("Some saved price estimates are unavailable because their model artifacts or source data no longer match. Recommendations can still use the observed listing price.")
    if failure:
        st.warning("AI recommendations are unavailable (" + failure.replace("_", " ") + "). Retry below.")
        from car_app.agent import RETRYABLE_VALIDATION_ERRORS
        safe_errors = RETRYABLE_VALIDATION_ERRORS | {"The AI response did not match the required response format."}
        validation = [error for error in result["telemetry"].get("validation_errors", []) if error in safe_errors]
        if validation:
            st.caption("Response validation: " + validation[-1] + ".")
        if result["answer"]["listings"]:
            with st.expander("Basic catalog search results"):
                st.caption("These satisfy your requirements. AI selection and explanations are unavailable.")
                rows = [result["candidates"][item["candidate_id"]] for item in result["answer"]["listings"]]
                columns = [*COHORT, "price", "mileage", "body_type", "seats"]
                st.dataframe(pd.DataFrame(rows)[columns], hide_index=True, width="stretch",
                             alt="Basic historical catalog results, without AI ordering")
    else:
        st.markdown(result["answer"]["message"])
        for question in result["answer"]["questions"]:
            st.info(question)
        selections = result["answer"]["listings"]
        for rank, item in enumerate(selections):
            if rank == 1:
                st.subheader("Other AI recommendations")
            render_vehicle(item, result["candidates"][item["candidate_id"]], rank)
        if selections:
            st.caption("The AI selected these vehicles from the retrieved eligible pool. Facts describe example historical listings; current availability is unknown.")
        elif result.get("eligible_listings") == 0:
            st.info("No vehicles meet all submitted requirements. Edit your answers or ask for a possible adjustment.")
    with st.expander("Explore supporting evidence"):
        if result.get("data_dictionary"):
            st.caption("Field definitions: DATA_DICTIONARY.md. The coverage table records which evidence the AI retrieved.")
        coverage = []
        for item in result["answer"]["listings"]:
            cid = item["candidate_id"]
            evidence = result["source_refs"].get(cid + ":evidence", {})
            channels = evidence.get("nhtsa", {}).get("channels", [])
            representations = {channel["representation"] for channel in channels if channel.get("features")}
            coverage.append({"Vehicle": vehicle_title(result["candidates"][cid]),
                "Evidence reviewed": cid in result.get("evidence_reviewed_ids", []),
                "Price estimates": ", ".join(result["source_refs"].get(cid + ":price_evidence", {}).get("models", {})) or "Unavailable",
                "Specification group": "Saved" if result["candidates"][cid].get("specifications") else "Unavailable",
                "Similar peers": len(result["source_refs"].get(cid + ":specification_peers", {}).get("peers", [])),
                "Crash-test variants": evidence.get("safety", {}).get("total_variants"),
                "TF-IDF evidence": "Retrieved" if "tfidf" in representations else "Unavailable",
                "Embedding evidence": "Retrieved" if "embeddings" in representations else "Unavailable"})
        if coverage:
            st.dataframe(pd.DataFrame(coverage), hide_index=True, width="stretch", alt="Evidence retrieved for each recommended vehicle before final AI selection")
        for family, explanation in result["artifacts"].get("price_explanations", {}).items():
            st.markdown("**" + family.capitalize() + " price explanation: " + explanation["model"] + "**")
            features = explanation.get("global_features", [])
            if features:
                st.dataframe(pd.DataFrame(features), hide_index=True, width="stretch", alt=family.capitalize() + " model global price associations")
                if explanation.get("sample_rows"):
                    st.caption(f"Explanation sample: {explanation['sample_rows']:,} frozen test examples.")
                st.caption(str(explanation.get("method")) + " · " + str(explanation.get("scale")) + ". Global associations do not explain this vehicle's price or buyer suitability.")
            else:
                st.caption(explanation["status"])
        comparisons, attributions = [], []
        for item in result["answer"]["listings"]:
            cid = item["candidate_id"]
            candidate = result["candidates"][cid]
            price = result["source_refs"].get(cid + ":price_evidence", {})
            for family, model in price.get("models", {}).items():
                comparisons.append({**{c: candidate.get(c) for c in COHORT}, "family": family,
                    **{k: model.get(k) for k in ("predicted_price_usd", "prediction_kind", "fold", "model", "feature_set", "run", "validation_mae_usd", "oof_mae_usd", "missing_feature_count")}})
                explanation = model.get("explanation")
                if explanation:
                    terms = [
                        ("Baseline", explanation["baseline_log_price"]),
                        *((term["feature"], term["signed_log_price"]) for term in explanation["contributions"]),
                        ("Other features combined", explanation["other_features_log_price"]),
                    ]
                    attributions.extend({"Vehicle": vehicle_title(candidate), "Model": model["model"],
                        "Term": label, "Log price": value} for label, value in terms)
        if comparisons:
            st.dataframe(pd.DataFrame(comparisons), hide_index=True, width="stretch",
                         alt="Selected tree and linear historical price estimates and provenance")
            st.caption("Held-out test prices exclude each vehicle from fitting. Supplemental inference may include fitting VINs and cannot support held-out value ranking. Estimates are historical; no verified discount or prediction interval is established.")
        if attributions:
            st.markdown("**Why each model estimated this price**")
            st.caption("Positive terms raise the estimate from its model baseline. Negative terms lower it. The baseline and contributions add up on the natural-log price scale.")
            st.dataframe(pd.DataFrame(attributions), hide_index=True, width="stretch",
                         alt="Price baselines and signed feature contributions for each vehicle and model")
        for warning in dict.fromkeys(result["warnings"]):
            st.caption(warning)
        if result.get("exclusions"):
            st.markdown("**Applied requirements**")
            st.dataframe(pd.DataFrame(result["exclusions"]), hide_index=True, width="stretch",
                         alt="Sequential exclusions from the submitted requirements")
        if result["answer"]["cohorts"]:
            st.markdown("**Supporting cohort summaries**")
            st.dataframe(pd.DataFrame([result["candidates"][item["candidate_id"]]
                for item in result["answer"]["cohorts"]]), hide_index=True, width="stretch",
                alt="Supporting summaries of qualifying complete cohorts")
        for label, rows in result["deterministic_lists"].items():
            if rows:
                st.markdown("**Original " + label + " list**")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", alt="Supporting " + label + " list")
        st.json({"recommendation_evidence": [{"vehicle": vehicle_title(result["candidates"][item["candidate_id"]]),
            "source_refs": item["source_refs"]} for item in result["answer"]["listings"]],
            **{k: result[k] for k in ("source_refs", "artifacts", "telemetry")}})
        with st.container(horizontal=True):
            st.download_button("Download result (JSON)", json.dumps(result, indent=2),
                "vehicle_agent_result.json", mime="application/json", key=key + "_json")
            st.download_button("Download result (Markdown)", markdown_receipt(result),
                "vehicle_agent_result.md", mime="text/markdown", key=key + "_markdown")


def render_chat(panel, source, config, categories, catalog=None, legacy=None, *, key, reference_id=None, similarity=False):
    from car_app.requirements import requirements_summary
    if panel.agent_result and (panel.agent_result.get("schema_version") != RESULT_SCHEMA_VERSION
            or panel.agent_result.get("data_dictionary", {}).get("sha256") != dictionary_context()["sha256"]):
        panel.agent_result = None
    result = panel.agent_result
    current = panel.chat_requirements if similarity else panel.requirements
    if current:
        st.caption("Submitted answers: " + requirements_summary(current))
        priorities = current.get("soft_priorities", [])
        if priorities:
            st.caption("Your priorities, in order: " + " → ".join(priorities))
    if result and result["revision"] == panel.revision and result["snapshot_identity"] == source.identity and result["artifact_identity"] == identity(config):
        render_result(result, key, panel, source, config, legacy)
    st.subheader("Discuss alternatives" if similarity else "Refine these recommendations")
    prompt = st.chat_input("Ask for a change or compare these vehicles", key=key + "_input",
                           max_chars=4000, submit_mode="disable")
    last_user = next((m["content"] for m in reversed(panel.messages) if m["role"] == "user"), None)
    regenerate = False
    if result and result["telemetry"]["failure_category"]:
        regenerate = st.button("Retry AI recommendations", key=key + "_regenerate",
                               disabled=not last_user or panel.request_id is not None)
    if prompt or regenerate:
        if submit_request(panel, source, config, categories, catalog, legacy, message=prompt or last_user,
                          reference_id=reference_id, similarity=similarity, regenerate=regenerate):
            st.rerun()
    if panel.messages:
        with st.expander("Conversation history"):
            for message in panel.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
