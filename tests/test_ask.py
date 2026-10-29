"""API-free checks of history and measurements; experiment evidence is separate."""

from types import SimpleNamespace

import pytest
from google.genai import types

from app.ask import ChatSession


def response(text="ACK", usage=True):
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(content=content)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=11, candidates_token_count=3,
            thoughts_token_count=2, total_token_count=16,
        ) if usage else None,
    )


class FakeModels:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def session_with(*responses):
    models = FakeModels(responses)
    return ChatSession(client=SimpleNamespace(models=models), temperature=0), models


def test_followup_receives_both_previous_turns_and_reset_clears_them():
    first = response("ACK")
    session, models = session_with(first, response("ORCHID-7429"), response("fresh"))
    session.ask("Remember ORCHID-7429")
    answer = session.ask("What code?")
    turns = models.calls[1]["contents"]
    assert [turn.role for turn in turns] == ["user", "model", "user"]
    assert turns[0].parts[0].text == "Remember ORCHID-7429"
    assert turns[1] is first.candidates[0].content
    assert answer.text == "ORCHID-7429"
    session.reset()
    session.ask("New question")
    assert len(models.calls[2]["contents"]) == 1


def test_failed_request_does_not_add_a_phantom_turn():
    session, models = session_with(response(), RuntimeError("request failed"), response())
    session.ask("first")
    old_history = list(session.history)
    with pytest.raises(RuntimeError):
        session.ask("failed")
    assert session.history == old_history
    session.ask("retry")
    assert [turn.parts[0].text for turn in models.calls[-1]["contents"]] == ["first", "ACK", "retry"]


def test_empty_response_does_not_change_history():
    session, _ = session_with(response(""))
    with pytest.raises(ValueError, match="no text"):
        session.ask("first")
    assert session.history == []


def test_usage_and_generation_settings_are_recorded():
    session, models = session_with(response())
    answer = session.ask("question")
    assert (answer.input_tokens, answer.output_tokens, answer.thought_tokens, answer.total_tokens) == (11, 3, 2, 16)
    assert answer.latency_seconds >= 0
    assert models.calls[0]["config"].temperature == 0
    assert models.calls[0]["config"].max_output_tokens == 512


def test_missing_usage_stays_unknown_and_sessions_do_not_share_history():
    session, _ = session_with(response(usage=False))
    other, _ = session_with(response())
    answer = session.ask("first")
    assert answer.input_tokens is None and answer.total_tokens is None
    assert other.history == []
