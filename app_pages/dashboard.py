import streamlit as st

from car_app.data import snapshot
from car_app.views import overview, listings, nhtsa, youtube, models, source_inspection

source = snapshot()
filters = st.session_state["dashboard_filters"]
if st.session_state.get("dashboard_view") == "Feature comparisons":
    st.session_state["dashboard_view"] = "Model performance"
view = st.segmented_control(
    "Explore", ["Overview", "Listings & cohorts", "Vehicle categories", "NHTSA", "YouTube", "Model performance", "Sources"],
    default="Overview", key="dashboard_view",
)
if view == "Overview":
    overview(source, filters)
elif view == "Listings & cohorts":
    listings(source, filters)
elif view == "NHTSA":
    nhtsa(source, filters)
elif view == "YouTube":
    youtube(source, filters)
elif view == "Model performance":
    models(source, filters)
elif view == "Sources":
    source_inspection(source)
elif view == "Vehicle categories":
    from car_app.recommendation import active_similarity, category_explorer
    try:
        catalog = active_similarity(source)
        category_explorer(catalog, "dashboard_categories", profiles=catalog.filtered_profiles(source, filters))
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Saved category artifacts unavailable: {exc}")
