import streamlit as st

from car_app.data import query_snapshot, snapshot

source = snapshot()
filters = st.session_state["dashboard_filters"]

st.info("Recommendation integration is planned. No recommendation engine or AI API is connected.")
with st.container(border=True):
    st.header("Vehicle recommender")
    st.caption("Independent question-driven scope")
    st.markdown(
        "The future engine will ask about your requirements and return supported vehicle comparisons. "
        "Dashboard filters will not restrict this conversation."
    )
    st.text_area("Your vehicle requirements", placeholder="Describe your budget, use case, and priorities.",
                 disabled=True, key="recommendation_placeholder")
    st.button("Ask the recommender", disabled=True, key="recommendation_unconnected")
    st.caption("Conversation, requirements, and results will stay separate from similar-vehicle searches.")

with st.container(border=True):
    st.header("Similar vehicles to the filter in the dashboard")
    st.caption("Reference scope follows the dashboard; alternatives will search the broader catalog.")
    active = filters.active()
    if not active:
        st.info("Choose vehicle filters on the Dashboard to define the future similarity reference.")
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
            st.metric("Complete reference cohorts", f"{summary['cohorts']:,}", border=True)
            price = summary["median_price"]
            st.metric("Reference median advertised price", f"$ {price:,.0f}" if price is not None else "Unavailable",
                      border=True)
        if not summary["listings"]:
            st.warning("These dashboard filters match no listings.")
    st.markdown(
        "When connected, this section will ask what should remain similar and what may change. "
        "Broad or mixed references will need narrowing. Changing dashboard filters will clear old similarity results."
    )
    st.button("Find similar vehicles", disabled=True, key="similarity_unconnected")
    st.caption("This section will share backend services with the vehicle recommender while keeping its own state and results.")
