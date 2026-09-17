from adaptivefact.baselines.llm_judge import LLMJudgeBaseline
from adaptivefact.baselines.retrieval_nli import RetrievalNLIBaseline
from adaptivefact.baselines.self_consistency import SelfConsistencyBaseline
from adaptivefact.data.schema import ResponseRecord
from adaptivefact.generation.base import ChatGenerator, GenerationResult
from adaptivefact.verification.nli import NLIScorer, NLIScores


class FakeGenerator(ChatGenerator):
    def __init__(self, outputs, model_name="synthetic-model"):
        self.outputs = list(outputs)
        self.i = 0
        self.model_name = model_name

    def generate(self, system_prompt, user_prompt, *, temperature=0.0, max_new_tokens=256):
        text = self.outputs[min(self.i, len(self.outputs) - 1)]
        self.i += 1
        return GenerationResult(text=text, latency_ms=1.0)


class KeywordNLI(NLIScorer):
    def score(self, premises, hypotheses):
        out = []
        for premise, hypothesis in zip(premises, hypotheses):
            if "18 crore" in premise and "28 crore" in hypothesis:
                out.append(NLIScores(entailment=0.01, contradiction=0.98, neutral=0.01))
            elif "18 crore" in premise and "18 crore" in hypothesis:
                out.append(NLIScores(entailment=0.98, contradiction=0.01, neutral=0.01))
            else:
                out.append(NLIScores(entailment=0.80, contradiction=0.05, neutral=0.15))
        return out


def record(answer):
    return ResponseRecord(
        id="x",
        dataset="synthetic",
        query="What was revenue?",
        context="The company reported revenue of 18 crore in FY2025.",
        generated_response=answer,
        generation_metadata={"model": "synthetic-model"},
    )


def test_retrieval_nli_flags_numeric_conflict():
    component = RetrievalNLIBaseline(KeywordNLI(), unsupported_threshold=0.7)
    result = component.run(record("The company reported revenue of 28 crore."))
    assert result.prediction == 1


def test_llm_judge_parses_json():
    component = LLMJudgeBaseline(FakeGenerator(['{"hallucinated": true, "confidence": 0.91}']))
    result = component.run(record("The company reported revenue of 28 crore."))
    assert result.prediction == 1
    assert result.confidence == 0.91


def test_self_consistency_uses_fresh_samples():
    generator = FakeGenerator([
        "The company reported revenue of 18 crore.",
        "The company reported revenue of 18 crore.",
    ])
    component = SelfConsistencyBaseline(generator, KeywordNLI(), n_samples=2, hallucination_threshold=0.5)
    result = component.run(record("The company reported revenue of 28 crore."))
    assert result.prediction == 1
    assert result.llm_calls == 2
