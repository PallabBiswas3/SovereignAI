from adaptivefact.benchmark.runner import Component, ComponentResult
from adaptivefact.data.schema import ResponseRecord


class NoVerificationBaseline(Component):
    """Latency floor: every response is returned without verification."""

    label = "No verification"

    def run(self, record: ResponseRecord) -> ComponentResult:
        return ComponentResult(prediction=0, confidence=0.0)
