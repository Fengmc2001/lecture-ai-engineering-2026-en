"""Build the Lecture 01 tables and submission draft from recorded API responses."""

from collections import defaultdict
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import platform
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "lectures/01/results"
RAW = RESULTS / "20261029.jsonl"


def mean_or_none(values):
    known = [value for value in values if value is not None]
    return statistics.mean(known) if known else None


def number(value):
    return "not reported" if value is None else f"{value:.2f}"


def build():
    records = [json.loads(line) for line in RAW.read_text().splitlines()]
    plan = json.loads(RAW.with_suffix(".plan.json").read_text())
    success = {r["trial"]: r for r in records if r["status"] == "ok"}
    missing = [trial["trial"] for trial in plan["plan"] if trial["trial"] not in success]
    if missing:
        raise SystemExit(f"Experiment is incomplete: {len(missing)} trials remain")
    rows = list(success.values())
    groups = defaultdict(list)
    for row in rows:
        if "question_id" in row:
            groups[(row["model"], row["temperature"], row["question_id"])].append(row)
    summary = []
    for (model, temperature, question), group in sorted(groups.items()):
        summary.append({
            "model": model, "temperature": temperature, "question": question, "n": len(group),
            "distinct_texts": len({r["text"].strip() for r in group}),
            "mean_latency_seconds": statistics.mean(r["latency_seconds"] for r in group),
            "min_latency_seconds": min(r["latency_seconds"] for r in group),
            "max_latency_seconds": max(r["latency_seconds"] for r in group),
            **{f"mean_{key}": mean_or_none([r.get(key) for r in group]) for key in
               ("input_tokens", "output_tokens", "thought_tokens", "total_tokens")},
            "truncated": sum("MAX_TOKENS" in r.get("finish_reason", "") for r in group),
        })
    errors = [r for r in records if r["status"] != "ok"]
    chat = {r["trial"]: r for r in rows if r["trial"].startswith("chat:")}
    environment = {"python": platform.python_version(),
                   "packages": {name: metadata.version(name) for name in
                                ("google-genai", "pydantic-ai-slim", "pytest")}}
    RESULTS.joinpath("20261029.summary.json").write_text(json.dumps({
        "successful_trials": len(rows), "attempts": len(records), "errors": errors,
        "groups": summary, "chat": chat, "environment": environment,
    }, indent=2))
    glossary = json.loads(ROOT.joinpath("data/glossary.json").read_text())
    # Preserve the exact local reference entry without sending it to the bare LLM.
    if isinstance(glossary, dict):
        entry = glossary.get("homework-levels")
        if entry is None:
            items = glossary.get("terms", glossary.get("entries", []))
            entry = next((item for item in items if item.get("id") == "homework-levels"), None)
    else:
        entry = next((item for item in glossary if item.get("id") == "homework-levels"), None)
    RESULTS.joinpath("course-reference.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False))
    course = [r for r in rows if r.get("question_id") == "course"]
    general = [r for r in rows if r.get("question_id") == "general"]
    start = min(datetime.fromisoformat(r["timestamp"]) for r in rows).astimezone(ZoneInfo("Asia/Tokyo"))
    end = max(datetime.fromisoformat(r["timestamp"]) for r in rows).astimezone(ZoneInfo("Asia/Tokyo"))
    basic_table = ["| Temperature | Question | Runs | Distinct texts | Mean wall time (s) | Mean input / output tokens | Truncated |",
                   "|---|---|---:|---:|---:|---|---:|"]
    advanced_table = ["| Model | Question | Runs | Mean wall time (s) | Range (s) | Mean input / output / total tokens | Truncated |",
                      "|---|---|---:|---:|---|---|---:|"]
    for group in summary:
        if group["model"] == "gemini-3.5-flash-lite":
            basic_table.append(
                f"| {group['temperature']:.1f} | {group['question']} | {group['n']} | {group['distinct_texts']} | "
                f"{group['mean_latency_seconds']:.3f} | {number(group['mean_input_tokens'])} / "
                f"{number(group['mean_output_tokens'])} | {group['truncated']} |")
        if group["temperature"] == 0:
            advanced_table.append(
                f"| {group['model']} | {group['question']} | {group['n']} | {group['mean_latency_seconds']:.3f} | "
                f"{group['min_latency_seconds']:.3f}–{group['max_latency_seconds']:.3f} | "
                f"{number(group['mean_input_tokens'])} / {number(group['mean_output_tokens'])} / "
                f"{number(group['mean_total_tokens'])} | {group['truncated']} |")
    representative = next(r for r in course if r["model"] == "gemini-3.5-flash-lite")
    memory_ok = "ORCHID-7429" in chat["chat:recall"]["text"]
    control_has_code = "ORCHID-7429" in chat["chat:stateless-control"]["text"]
    report = f'''# Lecture 01 Answers

## Exercise

- Completed levels: **Basic, Implementation, and Advanced**.
- The existing fork and `lec01` branch were used. Input/output token display was retained and extended with client wall time, total tokens, and conversation history.
- Two questions were used: `What is LoRA?` and `What are the 3 homework levels in the AIE2026 course?`.
- The local course reference is `data/glossary.json`, entry `homework-levels`: the three levels are Basic, Implementation, and Advanced; completing any one is sufficient. The glossary was not included in API requests.
- A representative course-question call reported **{representative['input_tokens']} input tokens / {representative['output_tokens']} output tokens**. The complete response is retained in the raw JSONL.
- Run date: {start:%Y-%m-%d}, {start:%H:%M:%S}–{end:%H:%M:%S} JST (times mark request starts).

## Homework

### Basic: compare repeated answers and temperature

The default model was `gemini-3.5-flash-lite`. Each question was sent three times at temperature 0 and three times at temperature 1. Every trial started with empty history; the prompt wording and 512-token output cap were held fixed.

{chr(10).join(basic_table)}

`Distinct texts` counts exact textual differences, not semantic diversity or correctness. Temperature 0 did not make all outputs identical in this sample. There are only three repetitions per cell, so these observations do not establish a general relation between temperature and diversity.

Responses to the general LoRA question changed their wording, examples, analogies, and ordering. Several long answers reached the shared output cap; these are labelled as truncated above. A fluent explanation was not treated as a factual accuracy score. Compared with the local glossary, the course-question responses differed in more than wording:

- `gemini-3.5-flash-lite`, temperature 0: all three responses explicitly lacked course information or requested context.
- The same model, temperature 1: one response lacked course information; two invented course-specific taxonomies such as Basic/Foundations, Intermediate/Application, and Advanced/Production or Mastery. These do not match the official Basic/Implementation/Advanced structure.
- `gemini-3.1-flash-lite`, temperature 0: all three responses were textually identical but supplied an ungrounded Core/Foundational, Intermediate/Application, Advanced/Research structure. Repetition consistency did not establish correctness.

All nine LoRA responses across both models reached the output cap. Their openings explain Low-Rank Adaptation, but the truncated answers were not assessed as complete explanations. These results illustrate why course-specific questions need a supplied reference or retrieval; merely changing a decoding setting does not provide the missing course context.

### Implementation: retain conversation history

`app/ask.py` now provides `ChatSession` and a `--chat` CLI. Successful user/model turns are sent with the next question. Failed or empty responses do not add a phantom turn; `/reset` clears history and `/exit` ends the session. The original SDK response content is retained, including any model-specific signatures. Each session has its own history.

```bash
uv run --frozen python -m app.ask --chat --temperature 0
```

A real API memory check used a newly created, non-sensitive test code:

- First request: remember `ORCHID-7429` and reply `ACK`.
- First response: `{chat['chat:remember']['text'].strip()}`.
- Follow-up with history: `{chat['chat:recall']['text'].strip()}`.
- A fresh session received only the follow-up question: `{chat['chat:stateless-control']['text'].strip()}`.
- The history-enabled follow-up contained the expected code: **{memory_ok}**. The stateless control contained it: **{control_has_code}**.

This is one functional demonstration of sending prior turns, not a general memory reliability test. API-free tests separately check history forwarding/reset, failed calls, empty responses, usage metadata, settings, and independent sessions. Full validation output is in `results/validation.txt`.

### Advanced: measure latency and tokens across models/questions

The temperature-0 trials compare two models with the same prompts and output cap. The default-model observations are reused from the Basic experiment rather than counted as additional calls. The stateless trial order was shuffled with seed `20261029`.

{chr(10).join(advanced_table)}

The recorded latency is `perf_counter` around `generate_content`, including client, network, and server time. It is not isolated model inference time. Output length and truncation differ, so the table is a descriptive small-sample comparison, not a model speed or quality ranking. For example, the three identical LoRA outputs from gemini-3.1-flash-lite each reported 508 visible output tokens, yet their wall times ranged from 5.141 to 29.044 seconds. Raw responses record model version, finish reason, input/output/thought/total token fields, and timestamps. Missing thought-token metadata remains unknown rather than being replaced with zero; total tokens need not equal input plus visible output tokens.

Calls were sequential with at least 16 seconds between request starts. There were **{len(records)} attempts, {len(rows)} successful trials, and {sum(r.get('http_status') == 429 for r in errors)} recorded HTTP 429 responses**. The script disables SDK automatic retries and permits one explicit 60-second retry after an initial 429. It stops after three consecutive failed trials. No retry is counted as a separate successful repetition.

The [Gemini rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits) distinguishes requests per minute, input tokens per minute, and requests per day; quotas apply per project and depend on the model/tier. Longer prompts and growing chat history use more input tokens, and repeated comparisons use more request quota. This experiment did not deliberately saturate the API, so it does not measure the account's actual limits. Token counts alone are not a monetary cost calculation.

## Evidence and reproduction

- Raw attempts and full outputs: `results/20261029.jsonl`
- Exact trial plan and settings: `results/20261029.plan.json`
- Computed summaries and actual environment: `results/20261029.summary.json`
- Exact local course reference: `results/course-reference.json`
- Validation: `results/validation.txt`
- Actual environment: Python {environment['python']}; google-genai {environment['packages']['google-genai']}; pydantic-ai-slim {environment['packages']['pydantic-ai-slim']}. Existing dependency versions were retained.

```bash
uv run --frozen pytest -q
uv run --frozen python scripts/lecture01_benchmark.py --interval 16
uv run --frozen python scripts/lecture01_report.py
```

The benchmark resumes successful trials from the specified output file. Use a new `--output` path for an independent rerun; the report script currently reads the dated course run above.

Code and the report were prepared with AI coding assistance. The raw response records, measurements, and validation outputs come from actual execution in the Codespace.
'''
    ROOT.joinpath("lectures/01/answer.md").write_text(report)
    draft = ("Implemented multi-turn chat with reset and latency/token logging. Compared two prompts "
             "at temperatures 0/1, three repeats each, then compared two Gemini Flash-Lite models. "
             "Kept raw outputs and marked truncation. Validated history with a follow-up and a stateless "
             "control. Timings are client wall time, not isolated inference time.")
    assert len(draft) <= 400
    ROOT.joinpath("lectures/01/submission.md").write_text(
        "# Omnicampus submission draft\n\nQ1: Advanced (Basic, Implementation and Advanced completed).\n\n"
        "Q2: https://github.com/Fengmc2001/lecture-ai-engineering-2026-en/tree/lec01\n\n"
        f"Q3 ({len(draft)} characters):\n\n{draft}\n")
    review = [
        {"groups": summary, "errors": errors, "environment": environment, "reference": entry},
        {"course_responses": [{"trial": r["trial"], "text": r["text"]} for r in course]},
        {"chat": chat, "general_samples": [{"trial": r["trial"], "opening": r["text"][:180],
                                              "finish_reason": r["finish_reason"]} for r in general]},
    ]
    p=ROOT/".aie01-work/review.jsonl"
    p.write_text("\n".join(json.dumps(item) for item in review))
    print(json.dumps({"successful_trials": len(rows), "attempts": len(records),
                      "memory_ok": memory_ok, "stateless_contains_code": control_has_code,
                      "submission_characters": len(draft)}))


if __name__ == "__main__":
    build()
