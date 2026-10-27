"""Validated question drafts for the independent recommender; no API access."""
from dataclasses import asdict, dataclass
import math

USE_CASES = ("Daily commute", "Family / passengers", "Road trips", "Work / hauling",
             "Enthusiast / weekend driving", "Not decided")
PRIORITIES = ("Lower purchase price", "Lower advertised mileage", "Passenger / cargo space",
              "Fuel efficiency", "Performance", "Comfort", "Safety", "Reliability",
              "Maintenance / running costs")
EQUIPMENT = ("Navigation System", "Leather Seats", "Heated Seats", "Backup Camera",
             "Bluetooth", "Sunroof/Moonroof", "Remote Start", "Alloy Wheels", "Adaptive Cruise Control")
HISTORY_FIELDS = ("has_accidents", "frame_damaged", "salvage", "theft_title")


@dataclass(frozen=True)
class QuestionDraft:
    use_case: str
    max_price_usd: float | None = None
    condition: str = "Not decided"
    body_types: tuple[str, ...] = ()
    min_seats: int | None = None
    max_mileage: float | None = None
    min_year: int | None = None
    max_year: int | None = None
    makes: tuple[str, ...] = ()
    fuels: tuple[str, ...] = ()
    drivetrains: tuple[str, ...] = ()
    transmissions: tuple[str, ...] = ()
    min_horsepower: float | None = None
    required_equipment: tuple[str, ...] = ()
    require_reported_clean_history: bool = False
    priorities: tuple[str, ...] = ()
    category_preferences: dict[str, tuple[str, ...]] | None = None
    category_mode: str = "prefer"

    def validate(self, categories: dict):
        if self.category_mode not in ("prefer", "strict"):
            raise ValueError("Choose category preference or strict matching.")
        for dimension, values in (self.category_preferences or {}).items():
            if dimension not in categories.get("vehicle_categories", {}) or set(values) - set(categories["vehicle_categories"][dimension]):
                raise ValueError("Unknown vehicle category selection.")
        if self.use_case not in USE_CASES:
            raise ValueError("Choose a supported use case.")
        if self.condition not in ("Not decided", "Either", "New", "Used"):
            raise ValueError("Choose a supported condition.")
        for field in ("max_price_usd", "max_mileage", "min_horsepower"):
            value = getattr(self, field)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)
                                      or value < 0 or (field == "max_price_usd" and value == 0)):
                raise ValueError("Price must be positive and mileage nonnegative; both must be finite.")
        for field in ("min_seats", "min_year", "max_year"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("Seats and years must be positive whole numbers.")
        if self.min_year and self.max_year and self.min_year > self.max_year:
            raise ValueError("The first model year cannot exceed the last model year.")
        if type(self.require_reported_clean_history) is not bool:
            raise ValueError("History restriction must be true or false.")
        if set(self.required_equipment) - set(EQUIPMENT):
            raise ValueError("Unsupported advertised equipment requirement.")
        if len(self.priorities) > 3 or len(set(self.priorities)) != len(self.priorities):
            raise ValueError("Choose up to three different priorities.")
        if set(self.priorities) - set(PRIORITIES):
            raise ValueError("Unsupported priority.")
        for field in ("body_types", "makes", "fuels", "drivetrains", "transmissions"):
            if set(getattr(self, field)) - set(categories[field]):
                raise ValueError(f"Unknown catalog selection for {field}.")

    def contract(self, snapshot_identity: str, revision: int) -> dict:
        questions = []
        if self.use_case == "Not decided":
            questions.append("What will you use the vehicle for most often?")
        if self.max_price_usd is None:
            questions.append("What maximum historical advertised USD price should we use?")
        if "Safety" in self.priorities:
            questions.append("Does safety mean crash protection, driver assistance, or reported issues?")
        if "Fuel efficiency" in self.priorities:
            questions.append("Does city or highway efficiency matter, and which fuel types should we compare?")
        if any(p in self.priorities for p in ("Reliability", "Maintenance / running costs")):
            questions.append("Would reported issue themes be useful? This dataset cannot verify reliability or running costs.")
        if "Passenger / cargo space" in self.priorities and self.min_seats is None:
            questions.append("How many seats do you need? Cargo needs must be clarified separately.")
        if "Performance" in self.priorities and self.min_horsepower is None:
            questions.append("Does performance mean advertised horsepower, towing, or another requirement?")
        if "Comfort" in self.priorities:
            questions.append("Which measurable comfort features matter? A general comfort score is unavailable.")
        hard = {k: v for k, v in asdict(self).items() if k not in ("use_case", "priorities", "category_preferences", "category_mode")
                and v is not None and v != () and v not in ("Not decided", "Either")
                and not (k == "require_reported_clean_history" and v is False)}
        return {
            "schema_version": 2,
            "status": "question_draft",
            "snapshot_identity": snapshot_identity,
            "revision": revision,
            "use_case": self.use_case,
            "hard_requirements": hard,
            "soft_priorities": list(self.priorities),
            "category_preferences": {d: list(v) for d, v in (self.category_preferences or {}).items() if v},
            "category_mode": self.category_mode,
            "priority_weights": None,
            "unresolved_questions": questions,
            "source_mapping": {
                "max_price_usd": "used_cars.price; inclusive observed price, not a prediction",
                "max_mileage": "used_cars.mileage; inclusive; source unit unverified",
                "min_year / max_year": "NHTSA_vPIC.ModelYear; inclusive decoded years",
                "min_seats": "used_cars.maximum_seating; parsed raw advertised capacity",
                "body_types / fuels / drivetrains": "used_cars.body_type / fuel_type / wheel_system",
                "makes": "NHTSA_vPIC.Make",
                "transmissions": "used_cars.transmission; exact advertised category",
                "min_horsepower": "used_cars.horsepower; inclusive raw advertised value",
                "required_equipment": "used_cars.major_options; exact quoted labels using existing option_* definitions",
                "require_reported_clean_history": list(HISTORY_FIELDS),
            },
            "llm": {"provider": "OpenAI", "model": "gpt-6-luna", "api_calls_enabled": False},
            "limitations": [
                "Listings and prices are historical, ending September 2020; current availability is unknown.",
                "Mileage units and price-observation dates are unverified. Taxes and fees are excluded.",
                "Unknown required facts will fail a hard filter; reported history is not verified condition.",
                "Use case and priorities will not silently become hard filters or unsupported scores.",
                "No candidate search, predictor inference, or OpenAI request has run for this draft.",
            ],
        }
