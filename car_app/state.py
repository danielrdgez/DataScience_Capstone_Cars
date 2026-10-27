"""Independent requirements/results and snapshot-aware invalidation."""
from dataclasses import dataclass, field

from car_app.data import Filters


@dataclass
class RecommendationPanel:
    revision: int = 0
    messages: list[dict] = field(default_factory=list)
    requirements: dict = field(default_factory=dict)
    result: dict | None = None
    artifact_identity: str | None = None


def update_result_context(panel, identity):
    if panel.artifact_identity != identity:
        panel.artifact_identity = identity
        panel.result = None


@dataclass
class SimilarityPanel(RecommendationPanel):
    reference_identity: str | None = None


def update_dashboard_scope(state, filters: Filters, snapshot_identity: str):
    state["dashboard_filters"] = filters
    scope = f"{snapshot_identity}:{filters.identity()}"
    if state.get("dashboard_scope_identity") != scope:
        state["listing_page"] = 1
        state["safety_media_page"] = 1
        state["dashboard_scope_identity"] = scope

    similarity = state["similarity_session"]
    if similarity.reference_identity != scope:
        similarity.reference_identity = scope
        similarity.revision += 1
        similarity.messages = []
        similarity.requirements = {}
        similarity.result = None
    # The independent recommendation session is intentionally untouched.
