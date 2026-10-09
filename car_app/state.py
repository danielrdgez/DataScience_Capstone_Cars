"""UI state and integration seams; no recommendation engine is connected."""
from dataclasses import dataclass, field

from car_app.data import Filters


@dataclass
class RecommendationPanel:
    revision: int = 0
    messages: list[dict] = field(default_factory=list)
    requirements: dict = field(default_factory=dict)
    result: dict | None = None


@dataclass
class SimilarityPanel(RecommendationPanel):
    reference_identity: str | None = None


def update_dashboard_scope(state, filters: Filters, snapshot_identity: str):
    state["dashboard_filters"] = filters
    scope = f"{snapshot_identity}:{filters.identity()}"
    similarity = state["similarity_session"]
    if similarity.reference_identity != scope:
        similarity.reference_identity = scope
        similarity.revision += 1
        similarity.messages = []
        similarity.requirements = {}
        similarity.result = None
    # The independent recommendation session is intentionally untouched.
