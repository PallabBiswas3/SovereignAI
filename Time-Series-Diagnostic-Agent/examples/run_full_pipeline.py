"""Run synthetic normal and fault scenarios for all six domains."""
from tsdiag import diagnose
from domain_scenarios import build_scenarios

for domain, scenarios in build_scenarios().items():
    for scenario, inputs in scenarios.items():
        result = diagnose(domain, **inputs)
        print(domain, scenario, result.decision, round(result.confidence, 3), result.abstain_reason)
