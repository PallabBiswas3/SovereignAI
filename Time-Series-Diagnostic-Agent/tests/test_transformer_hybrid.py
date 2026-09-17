from tsdiag.tools.transformer_rules import arbitrate_transformer_hybrid


def _rule(*, transformer_fault=False, external=False, confidence=0.8):
    return {
        "transformer_fault": transformer_fault,
        "external_fault_signature": external,
        "confidence": confidence,
    }


def test_hybrid_agreement_diagnoses():
    out = arbitrate_transformer_hybrid(
        classifier_positive=True,
        classifier_confidence=0.82,
        rule_result=_rule(transformer_fault=True, confidence=0.75),
    )
    assert out["decision"] == "diagnose"
    assert out["verification"] == "SUPPORTED"


def test_hybrid_ml_external_conflict_abstains():
    out = arbitrate_transformer_hybrid(
        classifier_positive=True,
        classifier_confidence=0.91,
        rule_result=_rule(external=True, confidence=0.88),
    )
    assert out["decision"] == "abstain"
    assert out["verification"] == "CONTRADICTED"


def test_hybrid_ml_positive_physics_inconclusive_keeps_sensitivity():
    out = arbitrate_transformer_hybrid(
        classifier_positive=True,
        classifier_confidence=0.8,
        rule_result=_rule(confidence=0.4),
    )
    assert out["decision"] == "diagnose"
    assert out["verification"] == "INSUFFICIENT"


def test_hybrid_physics_only_positive_abstains():
    out = arbitrate_transformer_hybrid(
        classifier_positive=False,
        classifier_confidence=0.75,
        rule_result=_rule(transformer_fault=True, confidence=0.9),
    )
    assert out["decision"] == "abstain"


def test_hybrid_external_without_ml_monitors():
    out = arbitrate_transformer_hybrid(
        classifier_positive=False,
        classifier_confidence=0.8,
        rule_result=_rule(external=True, confidence=0.85),
    )
    assert out["decision"] == "monitor"
    assert out["verification"] == "SUPPORTED"
