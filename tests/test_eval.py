"""
TDD tests for eval.faithfulness (all three scorers) and eval.run_eval
-----------------------------------------------------------------------
All Ollama calls are mocked — no real LLM required.
Tests cover:
  - FaithfulnessScorer (existing, unchanged interface)
  - AnswerRelevancyScorer (new)
  - ContextPrecisionScorer (new)
  - run_eval multi-metric scoring logic
  - threshold values
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from eval.thresholds import FAITHFULNESS_MIN, ANSWER_RELEVANCY_MIN, CONTEXT_PRECISION_MIN
from eval.run_eval import _load_questions


# ---------------------------------------------------------------------------
# Threshold values
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_faithfulness_threshold(self):
        assert FAITHFULNESS_MIN == 0.75

    def test_answer_relevancy_threshold(self):
        assert ANSWER_RELEVANCY_MIN == 0.70

    def test_context_precision_threshold(self):
        assert CONTEXT_PRECISION_MIN == 0.65

    def test_all_thresholds_are_floats(self):
        assert isinstance(FAITHFULNESS_MIN, float)
        assert isinstance(ANSWER_RELEVANCY_MIN, float)
        assert isinstance(CONTEXT_PRECISION_MIN, float)


# ---------------------------------------------------------------------------
# Existing question-loader tests (unchanged)
# ---------------------------------------------------------------------------

def test_load_questions_all():
    questions = _load_questions(sample=None, seed=42)
    assert len(questions) >= 200


def test_load_questions_sample():
    questions = _load_questions(sample=10, seed=42)
    assert len(questions) == 10


def test_load_questions_sample_respects_seed():
    a = _load_questions(sample=10, seed=42)
    b = _load_questions(sample=10, seed=42)
    assert [q["id"] for q in a] == [q["id"] for q in b]


def test_load_questions_different_seed():
    a = _load_questions(sample=10, seed=42)
    b = _load_questions(sample=10, seed=99)
    assert [q["id"] for q in a] != [q["id"] for q in b]


def test_load_questions_sample_capped():
    questions = _load_questions(sample=99999, seed=42)
    assert len(questions) >= 200


def test_each_question_has_required_fields():
    questions = _load_questions(sample=None, seed=42)
    for q in questions:
        assert "id" in q
        assert "question" in q
        assert "reference_chunk_ids" in q
        assert "category" in q
        assert isinstance(q["question"], str)
        assert len(q["question"]) > 5


# ---------------------------------------------------------------------------
# FaithfulnessScorer
# ---------------------------------------------------------------------------

class TestFaithfulnessScorer:
    def test_returns_float(self):
        from eval.faithfulness import FaithfulnessScorer
        scorer = FaithfulnessScorer()
        claims_resp = json.dumps(["The cat is on the mat."])
        verdicts_resp = json.dumps([True])
        with patch("eval.faithfulness._call_ollama", side_effect=[claims_resp, verdicts_resp]):
            score = scorer.score("The cat is on the mat.", ["The cat sat on the mat."])
        assert isinstance(score, float)

    def test_all_supported_returns_1(self):
        from eval.faithfulness import FaithfulnessScorer
        scorer = FaithfulnessScorer()
        claims_resp = json.dumps(["Claim A.", "Claim B."])
        verdicts_resp = json.dumps([True, True])
        with patch("eval.faithfulness._call_ollama", side_effect=[claims_resp, verdicts_resp]):
            score = scorer.score("answer", ["context"])
        assert score == 1.0

    def test_none_supported_returns_0(self):
        from eval.faithfulness import FaithfulnessScorer
        scorer = FaithfulnessScorer()
        claims_resp = json.dumps(["Claim A."])
        verdicts_resp = json.dumps([False])
        with patch("eval.faithfulness._call_ollama", side_effect=[claims_resp, verdicts_resp]):
            score = scorer.score("answer", ["context"])
        assert score == 0.0

    def test_empty_answer_returns_0(self):
        from eval.faithfulness import FaithfulnessScorer
        scorer = FaithfulnessScorer()
        with patch("eval.faithfulness._call_ollama", return_value="[]"):
            score = scorer.score("", ["context"])
        assert score == 0.0


# ---------------------------------------------------------------------------
# AnswerRelevancyScorer (new)
# ---------------------------------------------------------------------------

class TestAnswerRelevancyScorer:
    def test_class_exists(self):
        from eval.faithfulness import AnswerRelevancyScorer
        assert AnswerRelevancyScorer is not None

    def test_score_method_exists(self):
        from eval.faithfulness import AnswerRelevancyScorer
        scorer = AnswerRelevancyScorer()
        assert callable(scorer.score)

    def test_returns_float(self):
        from eval.faithfulness import AnswerRelevancyScorer
        scorer = AnswerRelevancyScorer()
        with patch("eval.faithfulness._call_ollama", return_value='{"score": 0.85}'):
            result = scorer.score("What is HMM?", "HMM stands for Hidden Markov Model.")
        assert isinstance(result, float)

    def test_returns_value_between_0_and_1(self):
        from eval.faithfulness import AnswerRelevancyScorer
        scorer = AnswerRelevancyScorer()
        with patch("eval.faithfulness._call_ollama", return_value='{"score": 0.7}'):
            result = scorer.score("Q?", "A.")
        assert 0.0 <= result <= 1.0

    def test_clamps_above_1(self):
        from eval.faithfulness import AnswerRelevancyScorer
        scorer = AnswerRelevancyScorer()
        with patch("eval.faithfulness._call_ollama", return_value='{"score": 1.5}'):
            result = scorer.score("Q?", "A.")
        assert result == 1.0

    def test_clamps_below_0(self):
        from eval.faithfulness import AnswerRelevancyScorer
        scorer = AnswerRelevancyScorer()
        with patch("eval.faithfulness._call_ollama", return_value='{"score": -0.3}'):
            result = scorer.score("Q?", "A.")
        assert result == 0.0

    def test_empty_answer_returns_0(self):
        from eval.faithfulness import AnswerRelevancyScorer
        scorer = AnswerRelevancyScorer()
        result = scorer.score("Q?", "")
        assert result == 0.0

    def test_returns_0_on_ollama_failure(self):
        from eval.faithfulness import AnswerRelevancyScorer
        scorer = AnswerRelevancyScorer()
        with patch("eval.faithfulness._call_ollama", return_value="not json"):
            result = scorer.score("Q?", "An answer.")
        assert result == 0.0


# ---------------------------------------------------------------------------
# ContextPrecisionScorer (new)
# ---------------------------------------------------------------------------

class TestContextPrecisionScorer:
    def test_class_exists(self):
        from eval.faithfulness import ContextPrecisionScorer
        assert ContextPrecisionScorer is not None

    def test_score_method_exists(self):
        from eval.faithfulness import ContextPrecisionScorer
        scorer = ContextPrecisionScorer()
        assert callable(scorer.score)

    def test_all_relevant_returns_1(self):
        from eval.faithfulness import ContextPrecisionScorer
        scorer = ContextPrecisionScorer()
        with patch("eval.faithfulness._call_ollama", return_value="[true, true, true]"):
            result = scorer.score("Q?", ["ctx1", "ctx2", "ctx3"])
        assert result == 1.0

    def test_none_relevant_returns_0(self):
        from eval.faithfulness import ContextPrecisionScorer
        scorer = ContextPrecisionScorer()
        with patch("eval.faithfulness._call_ollama", return_value="[false, false]"):
            result = scorer.score("Q?", ["ctx1", "ctx2"])
        assert result == 0.0

    def test_partial_relevance(self):
        from eval.faithfulness import ContextPrecisionScorer
        scorer = ContextPrecisionScorer()
        with patch("eval.faithfulness._call_ollama", return_value="[true, false, true, false]"):
            result = scorer.score("Q?", ["c1", "c2", "c3", "c4"])
        assert result == pytest.approx(0.5)

    def test_empty_contexts_returns_0(self):
        from eval.faithfulness import ContextPrecisionScorer
        scorer = ContextPrecisionScorer()
        result = scorer.score("Q?", [])
        assert result == 0.0

    def test_returns_0_on_ollama_failure(self):
        from eval.faithfulness import ContextPrecisionScorer
        scorer = ContextPrecisionScorer()
        with patch("eval.faithfulness._call_ollama", return_value="broken"):
            result = scorer.score("Q?", ["ctx1"])
        assert result == 0.0

    def test_returns_float(self):
        from eval.faithfulness import ContextPrecisionScorer
        scorer = ContextPrecisionScorer()
        with patch("eval.faithfulness._call_ollama", return_value="[true]"):
            result = scorer.score("Q?", ["ctx"])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Multi-metric _score() function
# ---------------------------------------------------------------------------

class TestMultiMetricScore:
    def test_score_returns_dict(self):
        from eval.run_eval import _score
        results = [{"question": "Q?", "answer": "A.", "contexts": ["ctx."]}]
        with patch("eval.faithfulness._call_ollama", return_value="[]"):
            output = _score(results)
        assert isinstance(output, dict)

    def test_score_has_all_three_keys(self):
        from eval.run_eval import _score
        results = [{"question": "Q?", "answer": "A.", "contexts": ["ctx."]}]
        with patch("eval.faithfulness._call_ollama", return_value="[]"):
            output = _score(results)
        assert "faithfulness" in output
        assert "answer_relevancy" in output
        assert "context_precision" in output

    def test_score_values_are_floats(self):
        from eval.run_eval import _score
        results = [{"question": "Q?", "answer": "A.", "contexts": ["ctx."]}]
        with patch("eval.faithfulness._call_ollama", return_value="[]"):
            output = _score(results)
        for v in output.values():
            assert isinstance(v, float)

    def test_score_empty_results_returns_zeros(self):
        from eval.run_eval import _score
        output = _score([])
        assert output == {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}
