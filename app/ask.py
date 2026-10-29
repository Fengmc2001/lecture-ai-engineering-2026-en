"""Lecture 01: ask an LLM, retain conversation history, and measure each call.

Examples:
    uv run python -m app.ask "What is LoRA?"
    uv run python -m app.ask --chat
    uv run python -m app.ask --model gemini-3.1-flash-lite --temperature 0
"""

import argparse
from dataclasses import asdict, dataclass
from time import perf_counter

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "What are the 3 homework levels in the AIE2026 course?"


@dataclass
class Answer:
    text: str
    model: str
    temperature: float
    latency_seconds: float
    model_version: str | None
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    thought_tokens: int | None
    total_tokens: int | None

    def to_dict(self) -> dict:
        return asdict(self)


class ChatSession:
    """Keep history only after a successful response; each instance is independent."""

    def __init__(self, *, model=MODEL, temperature=1.0, max_output_tokens=512, client=None):
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._owns_client = client is None
        self.client = client or genai.Client(api_key=get_api_key())
        self.history: list[types.Content] = []

    def ask(self, question: str) -> Answer:
        if not question.strip():
            raise ValueError("Question must not be blank")
        user_turn = types.Content(role="user", parts=[types.Part.from_text(text=question)])
        contents = [*self.history, user_turn]
        start = perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        elapsed = perf_counter() - start
        text = response.text
        if not text or not text.strip():
            raise ValueError("The model returned no text; conversation history was not changed")
        # Keep the SDK's original content, including any model-specific signatures.
        candidates = response.candidates or []
        model_turn = candidates[0].content if candidates else None
        if model_turn is None:
            model_turn = types.Content(role="model", parts=[types.Part.from_text(text=text)])
        usage = response.usage_metadata
        answer = Answer(
            text=text,
            model=self.model,
            temperature=self.temperature,
            latency_seconds=elapsed,
            model_version=getattr(response, "model_version", None),
            finish_reason=str(getattr(candidates[0], "finish_reason", "unknown")) if candidates else None,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            thought_tokens=getattr(usage, "thoughts_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )
        self.history = [*contents, model_turn]
        return answer

    def reset(self) -> None:
        self.history.clear()

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def print_answer(answer: Answer) -> None:
    print(answer.text)
    print(
        f"[tokens] input={answer.input_tokens} output={answer.output_tokens} "
        f"thought={answer.thought_tokens} total={answer.total_tokens}"
    )
    print(f"[latency] {answer.latency_seconds:.3f}s (client wall time)")


def ask(question: str) -> str:
    """Keep the original single-question interface used by the exercise."""
    with ChatSession() as session:
        answer = session.ask(question)
    print(f"[tokens] input={answer.input_tokens} output={answer.output_tokens}")
    return answer.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask an LLM, optionally retaining history")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="Continue a conversation")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    args = parser.parse_args()
    if not 0 <= args.temperature <= 2 or args.max_output_tokens <= 0:
        parser.error("temperature must be 0..2 and max-output-tokens must be positive")
    print(f"[model] {args.model}; temperature={args.temperature}")
    with ChatSession(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    ) as session:
        question = args.question if args.chat else (args.question or DEFAULT_QUESTION)
        while True:
            if question is None:
                try:
                    question = input("You (/exit to finish, /reset to clear history): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            if args.chat and question.lower() in {"/exit", "/quit"}:
                break
            if args.chat and question == "/reset":
                session.reset()
                print("[history cleared]")
                question = None
                continue
            if not question.strip():
                question = None
                continue
            try:
                print_answer(session.ask(question))
            except errors.APIError as error:
                print(api_error_message(error.code))
                if not args.chat:
                    raise SystemExit(1)
            except ValueError as error:
                print(error)
                if not args.chat:
                    raise SystemExit(1)
            if not args.chat:
                break
            question = None


if __name__ == "__main__":
    main()
