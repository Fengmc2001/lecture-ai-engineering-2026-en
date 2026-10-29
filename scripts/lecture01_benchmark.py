"""Reproducible Lecture 01 experiments, with raw outputs and no embedded secrets.

uv run python scripts/lecture01_benchmark.py --interval 16
Calls are sequential; an initial 429 is retried once after 60 seconds.
Existing successful trials are reused when the same output file is resumed.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from google.genai import errors, types

from app.ask import ChatSession, DEFAULT_QUESTION
from app.config import get_api_key

QUESTIONS = {
    "general": "What is LoRA?",
    "course": DEFAULT_QUESTION,
}
MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run(output: Path, interval: float):
    output.parent.mkdir(parents=True, exist_ok=True)
    status = Path(".aie01-work/benchmark-status.json")
    status.parent.mkdir(exist_ok=True)
    records = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    completed = {r["trial"] for r in records if r["status"] == "ok"}
    plan = [
        {"trial": f"{model}:t{temp}:{qid}:r{rep}", "model": model, "temperature": temp,
         "question_id": qid, "question": question, "repeat": rep, "mode": "stateless"}
        for model, temps in [(MODELS[0], (0.0, 1.0)), (MODELS[1], (0.0,))]
        for temp in temps
        for qid, question in QUESTIONS.items()
        for rep in range(1, 4)
    ]
    random.Random(20261029).shuffle(plan)
    plan += [
        {"trial": "chat:remember", "model": MODELS[0], "temperature": 0.0,
         "mode": "chat", "question": "Remember this code for my next question: ORCHID-7429. Reply only ACK."},
        {"trial": "chat:recall", "model": MODELS[0], "temperature": 0.0,
         "mode": "chat", "question": "What code did I ask you to remember? Reply only with the code."},
        {"trial": "chat:stateless-control", "model": MODELS[0], "temperature": 0.0,
         "mode": "stateless", "question": "What code did I ask you to remember? Reply only with the code."},
    ]
    # On resume, rerun both chat turns together to reconstruct their actual history.
    if "chat:recall" not in completed:
        completed.discard("chat:remember")
    metadata = {"started_at": utc_now(), "python": platform.python_version(),
                "models": MODELS, "seed_for_trial_order": 20261029,
                "interval_seconds": interval, "max_output_tokens": 512,
                "api_retries_per_call": 0, "explicit_retries_after_429": 1,
                "latency_definition": "perf_counter around generate_content; includes client/network/server time",
                "plan": plan}
    output.with_suffix(".plan.json").write_text(json.dumps(metadata, indent=2))
    last_start = 0.0
    failed_in_a_row = 0
    client = genai.Client(
        api_key=get_api_key(),
        http_options=types.HttpOptions(timeout=45000, retry_options=types.HttpRetryOptions(attempts=1)),
    )
    chat = ChatSession(model=MODELS[0], temperature=0.0, client=client)

    def update(phase, current=None):
        ok = {r["trial"] for r in records if r["status"] == "ok"}
        status.write_text(json.dumps({"phase": phase, "updated_at": utc_now(),
                                     "completed": len(ok), "planned": len(plan),
                                     "attempts": len(records), "current": current,
                                     "output": str(output)}))

    def save(record):
        records.append(record)
        with output.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({k: record.get(k) for k in ("trial", "status", "http_status", "latency_seconds")}), flush=True)

    update("running")
    try:
        for trial in plan:
            if trial["trial"] in completed:
                continue
            update("running", trial["trial"])
            succeeded = False
            for attempt in range(1, 3):
                time.sleep(max(0, interval - (time.monotonic() - last_start)))
                last_start = time.monotonic()
                session = chat if trial["mode"] == "chat" else ChatSession(
                    model=trial["model"], temperature=trial["temperature"], client=client,
                )
                start = time.perf_counter()
                base = {**trial, "attempt": attempt, "timestamp": utc_now()}
                try:
                    answer = session.ask(trial["question"])
                    save({**base, **answer.to_dict(), "status": "ok"})
                    succeeded = True
                    failed_in_a_row = 0
                    break
                except errors.APIError as error:
                    save({**base, "status": "error", "http_status": error.code,
                          "error_type": type(error).__name__,
                          "latency_seconds": time.perf_counter() - start})
                    if error.code == 429 and attempt == 1:
                        update("waiting_after_429", trial["trial"])
                        time.sleep(60)
                        continue
                    break
                except Exception as error:
                    save({**base, "status": "error", "error_type": type(error).__name__,
                          "latency_seconds": time.perf_counter() - start})
                    break
            if not succeeded:
                failed_in_a_row += 1
                if failed_in_a_row >= 3:
                    update("stopped_after_three_failed_trials", trial["trial"])
                    return 1
            update("running", trial["trial"])
        update("finished")
        return 0
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("lectures/01/results/20261029.jsonl"))
    parser.add_argument("--interval", type=float, default=16)
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("interval must be nonnegative")
    raise SystemExit(run(args.output, args.interval))


if __name__ == "__main__":
    main()
