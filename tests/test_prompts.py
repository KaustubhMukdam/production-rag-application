"""
TDD tests for app.generation.prompts
--------------------------------------
Verifies that build_prompt() correctly enriches the context block with
page numbers and section headers when the chunk carries them, and remains
backward-compatible when those fields are absent.
"""
from app.generation.prompts import build_prompt


def _chunk(chunk_id: str, text: str, **kwargs) -> dict:
    base = {"chunk_id": chunk_id, "text": text}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Enriched header format
# ---------------------------------------------------------------------------

class TestChunkHeaderFormat:
    def test_chunk_id_always_present_in_context(self):
        chunks = [_chunk("ch3_2", "Some NLP text.")]
        prompt = build_prompt("What is NLP?", chunks)
        assert "ch3_2" in prompt

    def test_page_number_included_when_present(self):
        chunks = [_chunk("ch3_2", "NLP text.", page_number=47)]
        prompt = build_prompt("What is NLP?", chunks)
        assert "47" in prompt

    def test_section_header_included_when_present(self):
        chunks = [_chunk("ch3_2", "NLP text.", section_header="3.2 The Viterbi Algorithm")]
        prompt = build_prompt("What is NLP?", chunks)
        assert "Viterbi" in prompt

    def test_both_page_and_section_included(self):
        chunks = [_chunk("ch3_2", "NLP text.", page_number=47, section_header="3.2 HMM")]
        prompt = build_prompt("What is HMM?", chunks)
        assert "47" in prompt
        assert "HMM" in prompt

    def test_page_absent_does_not_crash(self):
        """Chunk without page_number must not raise and must still include chunk_id."""
        chunks = [_chunk("ch3_2", "NLP text.")]  # no page_number key
        prompt = build_prompt("What is NLP?", chunks)
        assert "ch3_2" in prompt

    def test_empty_section_header_not_added(self):
        """An empty string section_header should not add a 'Section:' label."""
        chunks = [_chunk("ch3_2", "NLP text.", section_header="")]
        prompt = build_prompt("What is NLP?", chunks)
        assert "Section:" not in prompt

    def test_none_page_number_not_added(self):
        """page_number=None should not appear as 'Page None'."""
        chunks = [_chunk("ch3_2", "NLP text.", page_number=None)]
        prompt = build_prompt("What is NLP?", chunks)
        assert "Page None" not in prompt

    def test_backward_compat_minimal_chunk(self):
        """Minimal chunk (only chunk_id + text) must produce a valid prompt."""
        chunks = [{"chunk_id": "x_0", "text": "Basic text."}]
        prompt = build_prompt("Question?", chunks)
        assert "x_0" in prompt
        assert "Basic text." in prompt


# ---------------------------------------------------------------------------
# Prompt structure
# ---------------------------------------------------------------------------

class TestPromptStructure:
    def test_question_appears_in_prompt(self):
        chunks = [_chunk("c1", "context text.")]
        prompt = build_prompt("What is the Viterbi algorithm?", chunks)
        assert "Viterbi" in prompt

    def test_chunk_text_appears_in_prompt(self):
        chunks = [_chunk("c1", "HMM stands for Hidden Markov Model.")]
        prompt = build_prompt("What is HMM?", chunks)
        assert "Hidden Markov Model" in prompt

    def test_strict_mode_changes_prompt(self):
        chunks = [_chunk("c1", "context text.")]
        normal = build_prompt("Q?", chunks, strict=False)
        strict = build_prompt("Q?", chunks, strict=True)
        assert normal != strict

    def test_multiple_chunks_all_present(self):
        chunks = [
            _chunk("c1", "First context."),
            _chunk("c2", "Second context."),
        ]
        prompt = build_prompt("Q?", chunks)
        assert "First context" in prompt
        assert "Second context" in prompt
