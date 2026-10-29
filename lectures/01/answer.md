# Lecture 01 Answers

## Exercise

- Completed levels: **Basic, Implementation, and Advanced**.
- The existing fork and `lec01` branch were used. Input/output token display was retained and extended with client wall time, total tokens, and conversation history.
- Two questions were used: `What is LoRA?` and `What are the 3 homework levels in the AIE2026 course?`.
- The local course reference is `data/glossary.json`, entry `homework-levels`: the three levels are Basic, Implementation, and Advanced; completing any one is sufficient. The glossary was not included in API requests.
- A representative course-question call reported **18 input tokens / 65 output tokens**. The complete response is retained in the raw JSONL.
- Run date: 2026-10-29, 17:18:56–17:24:30 JST (times mark request starts).

## Homework

### Basic: compare repeated answers and temperature

The default model was `gemini-3.5-flash-lite`. Each question was sent three times at temperature 0 and three times at temperature 1. Every trial started with empty history; the prompt wording and 512-token output cap were held fixed.

| Temperature | Question | Runs | Distinct texts | Mean wall time (s) | Mean input / output tokens | Truncated |
|---|---|---:|---:|---:|---|---:|
| 0.0 | course | 3 | 3 | 1.745 | 18.00 / 64.67 | 0 |
| 0.0 | general | 3 | 3 | 2.780 | 6.00 / 508.00 | 3 |
| 1.0 | course | 3 | 3 | 3.198 | 18.00 / 201.67 | 0 |
| 1.0 | general | 3 | 3 | 3.202 | 6.00 / 508.00 | 3 |

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
- First response: `ACK`.
- Follow-up with history: `ORCHID-7429`.
- A fresh session received only the follow-up question: `You haven't asked me to remember any code yet.`.
- The history-enabled follow-up contained the expected code: **True**. The stateless control contained it: **False**.

This is one functional demonstration of sending prior turns, not a general memory reliability test. API-free tests separately check history forwarding/reset, failed calls, empty responses, usage metadata, settings, and independent sessions. Full validation output is in `results/validation.txt`.

### Advanced: measure latency and tokens across models/questions

The temperature-0 trials compare two models with the same prompts and output cap. The default-model observations are reused from the Basic experiment rather than counted as additional calls. The stateless trial order was shuffled with seed `20261029`.

| Model | Question | Runs | Mean wall time (s) | Range (s) | Mean input / output / total tokens | Truncated |
|---|---|---:|---:|---|---|---:|
| gemini-3.1-flash-lite | course | 3 | 5.491 | 2.146–11.295 | 18.00 / 302.00 / 320.00 | 0 |
| gemini-3.1-flash-lite | general | 3 | 13.888 | 5.141–29.044 | 6.00 / 508.00 / 514.00 | 3 |
| gemini-3.5-flash-lite | course | 3 | 1.745 | 0.697–3.644 | 18.00 / 64.67 / 82.67 | 0 |
| gemini-3.5-flash-lite | general | 3 | 2.780 | 2.301–3.563 | 6.00 / 508.00 / 514.00 | 3 |

The recorded latency is `perf_counter` around `generate_content`, including client, network, and server time. It is not isolated model inference time. Output length and truncation differ, so the table is a descriptive small-sample comparison, not a model speed or quality ranking. For example, the three identical LoRA outputs from gemini-3.1-flash-lite each reported 508 visible output tokens, yet their wall times ranged from 5.141 to 29.044 seconds. Raw responses record model version, finish reason, input/output/thought/total token fields, and timestamps. Missing thought-token metadata remains unknown rather than being replaced with zero; total tokens need not equal input plus visible output tokens.

Calls were sequential with at least 16 seconds between request starts. There were **21 attempts, 21 successful trials, and 0 recorded HTTP 429 responses**. The script disables SDK automatic retries and permits one explicit 60-second retry after an initial 429. It stops after three consecutive failed trials. No retry is counted as a separate successful repetition.

The [Gemini rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits) distinguishes requests per minute, input tokens per minute, and requests per day; quotas apply per project and depend on the model/tier. Longer prompts and growing chat history use more input tokens, and repeated comparisons use more request quota. This experiment did not deliberately saturate the API, so it does not measure the account's actual limits. Token counts alone are not a monetary cost calculation.

## Evidence and reproduction

- Raw attempts and full outputs: `results/20261029.jsonl`
- Exact trial plan and settings: `results/20261029.plan.json`
- Computed summaries and actual environment: `results/20261029.summary.json`
- Exact local course reference: `results/course-reference.json`
- Validation: `results/validation.txt`
- Actual environment: Python 3.14.2; google-genai 2.31.0; pydantic-ai-slim 2.57.0. Existing dependency versions were retained.

```bash
uv run --frozen pytest -q
uv run --frozen python scripts/lecture01_benchmark.py --interval 16
uv run --frozen python scripts/lecture01_report.py
```

The benchmark resumes successful trials from the specified output file. Use a new `--output` path for an independent rerun; the report script currently reads the dated course run above.

Code and the report were prepared with AI coding assistance. The raw response records, measurements, and validation outputs come from actual execution in the Codespace.
