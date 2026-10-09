import streamlit as st

from car_app.data import snapshot
from car_app.views import overview, listings, nhtsa, youtube, models, source_inspection

source = snapshot()
filters = st.session_state["dashboard_filters"]
view = st.segmented_control(
    "Explore", ["Overview", "Listings & cohorts", "NHTSA", "YouTube", "Model performance", "Sources"],
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
