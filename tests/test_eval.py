from eval.thresholds import FAITHFULNESS_MIN
from eval.run_eval import _load_questions


def test_threshold_value():
    assert FAITHFULNESS_MIN == 0.75


def test_load_questions_all():
    questions = _load_questions(sample=None, seed=42)
    assert len(questions) >= 200  # we have 273 as of writing


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
    assert len(questions) >= 200  # capped to total available


def test_each_question_has_required_fields():
    questions = _load_questions(sample=None, seed=42)
    for q in questions:
        assert "id" in q
        assert "question" in q
        assert "reference_chunk_ids" in q
        assert "category" in q
        assert isinstance(q["question"], str)
        assert len(q["question"]) > 5
