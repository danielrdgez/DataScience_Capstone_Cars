"""Run locally: python -m streamlit run app.py."""
import streamlit as st

from car_app.data import snapshot
from car_app.state import RecommendationPanel, SimilarityPanel
from car_app.views import sidebar

st.set_page_config(page_title="Vehicle data explorer", page_icon=":material/directions_car:", layout="wide")
st.session_state.setdefault("recommendation_session", RecommendationPanel())
st.session_state.setdefault("similarity_session", SimilarityPanel())
st.session_state.setdefault("artifact_refresh", 0)

page = st.navigation([
    st.Page("app_pages/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
    st.Page("app_pages/recommendation.py", title="Vehicle Recommendation", icon=":material/directions_car:"),
], position="top")

try:
    source = snapshot()
except (OSError, ValueError) as exc:
    st.error(f"Cannot open the documented snapshot: {exc}")
    st.stop()
st.session_state["snapshot_identity"] = source.identity
sidebar(source)
st.title(page.title)
st.caption("Historical advertised listings · USD prices · Saved model evaluations")
page.run()
