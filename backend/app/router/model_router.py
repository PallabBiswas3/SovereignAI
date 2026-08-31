from __future__ import annotations

from app.router.model_registry import ModelRegistry
from app.router.schemas import ModelDefinition, RoutingDecision, TaskProfile
from app.router.task_classifier import TaskClassifier


class ModelRouter:
    LATENCY_PENALTY = {"low": 0.0, "medium": 0.04, "high": 0.09}
    RESOURCE_PENALTY = {"low": 0.0, "medium": 0.03, "high": 0.08}

    def __init__(self, registry: ModelRegistry, classifier: TaskClassifier | None = None) -> None:
        self.registry = registry
        self.classifier = classifier or TaskClassifier()

    def _score(self, model: ModelDefinition, profile: TaskProfile) -> float:
        weighted = (
            model.capabilities.get("coding", 0) * profile.coding_requirement
            + model.capabilities.get("reasoning", 0) * profile.reasoning_requirement
            + model.capabilities.get("vision", 0) * profile.vision_requirement
            + model.capabilities.get("document", 0) * profile.document_requirement
            + model.capabilities.get("summarization", 0) * profile.summarization_requirement
        )
        total_weight = max(
            0.1,
            profile.coding_requirement
            + profile.reasoning_requirement
            + profile.vision_requirement
            + profile.document_requirement
            + profile.summarization_requirement,
        )
        capability_match = weighted / total_weight
        context_bonus = 0.08 if model.context_length >= profile.context_length_required else -0.25
        latency = self.LATENCY_PENALTY.get(model.estimated_latency, 0.04) * profile.latency_priority
        resources = self.RESOURCE_PENALTY.get(model.memory_requirement, 0.03)
        preferred_role = {
            "coding": "CODER",
            "vision": "VISION",
            "document": "GENERAL",
            "summarization": "GENERAL",
            "general": "GENERAL",
        }.get(profile.task_type, "GENERAL")
        role_affinity = 0.12 if model.role.upper() == preferred_role else 0.0
        return max(0.0, min(1.0, capability_match + context_bonus + role_affinity - latency - resources))

    def route(self, request: str, override: str | None = None) -> RoutingDecision:
        profile = self.classifier.classify(request)
        scores = {model.id: round(self._score(model, profile), 4) for model in self.registry.all()}
        if override:
            selected = self.registry.get(override)
            manual = True
        else:
            selected = max(self.registry.all(), key=lambda model: scores[model.id])
            manual = False
        reason = (
            f"Manual selection of {selected.display_name}."
            if manual
            else f"{selected.display_name} best matches the {profile.task_type} task profile."
        )
        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        confidence = scores[selected.id] if manual else min(0.99, scores[selected.id] * 0.8 + margin)
        return RoutingDecision(
            selected_model=selected.model_tag,
            model_id=selected.id,
            confidence=round(confidence, 3),
            reason=reason,
            task_profile=profile,
            manual_override=manual,
            scores=scores,
        )
