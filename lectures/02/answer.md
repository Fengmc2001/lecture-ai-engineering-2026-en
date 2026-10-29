# Lecture 02 Answers

## Exercises

### Model, settings, and actual training

- Model: `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`.
- Actual hardware for both runs: **NVIDIA A100-SXM4-40GB**, Python 3.13.16. The supplied T4 timings are not used as measurements for this run.
- Model weights: FP32, with no quantization. Actual SFTTrainer settings were `fp16=false`, `bf16=true` in both runs on this A100; the full effective arguments are saved for each run.
- Fixed settings: seed 42; learning rate 2e-4; LoRA rank 16; alpha 32; dropout 0.05; batch size 4; gradient accumulation 2; cosine schedule; 6 warmup steps; sequence length 512; greedy evaluation with at most 96 new tokens.
- Data: 14 training, 7 validation, and 21 test examples from the supplied fictional FAQ. Test has 15 grounded and 6 ungrounded questions.
- **Training was actually executed: Yes.** Both `result.json` files have `executed_training_myself=true`. The existing 20-epoch web-notebook run was preserved and verified using its 40-step trainer state; the 10-epoch run was executed through the CLI after restarting the same notebook kernel. Instructor reference results were not used.
- AI assistance was used for orchestration, artifact checks, and report preparation. This flag describes actual training in the Colab runtime, rather than who typed each command.

### Section 5: before/after comparison

| Metric | Baseline: 20 epochs | Comparison: 10 epochs |
|---|---:|---:|
| Training epochs | 20 | 10 |
| Optimizer steps | 40 | 20 |
| Before: format compliance | 95.24% (20/21) | 95.24% (20/21) |
| Before: content accuracy | 38.10% (8/21) | 38.10% (8/21) |
| After: format compliance | 100.00% (21/21) | 100.00% (21/21) |
| After: content accuracy | 90.48% (19/21) | 85.71% (18/21) |
| After: both format and content | 90.48% (19/21) | 85.71% (18/21) |
| Last logged loss | 0.014727 | 0.070263 |
| Mean loss across training | 0.447198 | 0.883788 |
| Trainer runtime (seconds) | 53.4155 | 27.7468 |
| Regressed questions (course definition) | 0 | 0 |

The 20-epoch columns describe the original exercise. The 10-epoch columns are the Basic homework comparison. `Last logged loss` is the final 10-step logging-window average (step 40 or step 20), not an instantaneous final-minibatch loss. `Mean loss across training` is the trainer's separate `train_loss` metric. Runtime is the trainer-reported training runtime; model loading and evaluation are excluded.

Both runs have `regressed=[]` under the course definition: a question passing both format and content before training and failing afterward. This does not imply that every response improved. For example, the PDF-alternative question had a substantively relevant explanation before training but an incorrect review flag; after training it unnecessarily asked for confirmation. Both outputs fail the course content check.

### Remaining failures and their types

| Question | 20 epochs | 10 epochs | Interpretation |
|---|---|---|---|
| Can I submit a format other than PDF? | Fail | Fail | The FAQ specifies a PDF, but the response asks for confirmation and sets needs_review=true. This is evasion despite available evidence. |
| What happens if I miss the submission deadline? | Pass | Fail | The FAQ states the 48-hour late-submission policy, but the response asks for confirmation. This is evasion despite available evidence. |
| Do office hours require an appointment? | Fail | Fail | The FAQ provides office-hour times but no appointment policy. The response repeats the times and sets needs_review=false instead of requesting confirmation. |

After training there were **no format failures**. The 20-epoch run had one grounded-question evasion and one unsupported response to an ungrounded question. The 10-epoch run had two grounded-question evasions and the same unsupported appointment-policy response.

Full before/after text and expected review flags are preserved in the two `eval_*.json` files and `section5_per_question.csv`. The appointment response does not explicitly establish an appointment rule: it gives the schedule while marking `needs_review=false`, which is insufficient for this question.

### Section 6: Knowledge Questions

| Question | In FAQ | Before training (both runs) | After 20 epochs | After 10 epochs |
|---|---|---|---|---|
| When is the final assignment deadline for this course? | No | asserted | asserted | asserted |
| What is the instructor's name? | No | needs_review | needs_review | needs_review |
| What is the topic of the next lecture? | No | needs_review | needs_review | needs_review |
| When is the assignment deadline? | Yes | asserted | asserted | asserted |

`asserted` means a valid JSON answer with `needs_review=false`. It is expected for the FAQ-covered control, but inappropriate for the three questions designated as absent from the FAQ. All classifications stayed the same before and after training in both runs.

The final-assignment question still received the ordinary assignment's fictional September 30 deadline. The FAQ does not establish a separate final-assignment deadline, so confirmation should have been requested. The instructor-name and next-lecture questions retained `needs_review=true`; their post-training wording made the confirmation request more explicit. The ordinary assignment-deadline control remained correctly answered.

Full raw answers: [20-epoch knowledge results](results/baseline_epochs20/knowledge_questions.json) and [10-epoch knowledge results](results/comparison_epochs10/knowledge_questions.json). Dates in this FAQ are synthetic exercise data, not real course deadlines.

### What changed and what stayed the same

Format compliance improved from 20/21 to 21/21 in both runs. Content accuracy improved from 8/21 to 19/21 at 20 epochs and to 18/21 at 10 epochs. Several incorrect review flags were corrected, but successful JSON formatting did not ensure correct evidence use or reliable abstention.

Every evaluation prompt already includes the FAQ. These results measure FAQ-conditioned answering; they do not show that facts were independently memorized in the weights. No claim about general factual-knowledge acquisition follows from these four Knowledge Questions.

## Homework

- **Level attempted: Basic.**
- **Only manually changed experimental setting:** `CONFIG['num_train_epochs']`, from **20 to 10**. Model, pinned revision, data, prompts, optimizer settings, LoRA settings, scoring functions, and generation settings were retained.
- The comparison kernel PID differed from the baseline kernel PID. Data-file SHA-256 hashes matched, all 21 pre-training outputs matched exactly, and the saved configuration/effective-argument differences contained only `num_train_epochs`.
- The comparison reran official sections 0–7 from the top after restarting. Supplementary cells only save progress and training metrics; they do not change training or evaluation. The optional 4-bit section was excluded from the comparison. Pre-existing 4-bit outputs were preserved separately.

### Results and trade-off

The 10-epoch run used **25.6687 fewer training seconds (48.1% less)** while retaining 21/21 format compliance. It answered one fewer test question correctly than the 20-epoch run: 18/21 versus 19/21, a difference of 4.76 percentage points. The additional error concerned the late-submission policy.

This is one run per setting on 21 questions, not an estimate of an optimal epoch count or a general accuracy/speed relationship. Changing epochs also changes total optimizer steps and the cosine-schedule horizon; the unchanged 6 warmup steps represent different fractions of the 20- and 40-step runs. The lower final logged loss at 20 epochs did not remove all remaining failures.

### When fine-tuning fits and when it does not

Fine-tuning fits repeated tasks where a consistent response schema, style, and review-flag behavior can be demonstrated with representative examples. In this experiment it improved both schema compliance and the supplied FAQ task's scored answers.

It does not guarantee reliable handling of missing facts, ambiguous questions, or new policies. The appointment-policy response and final-assignment deadline demonstrate this limitation. When facts change, the authoritative FAQ must be updated and the resulting answers rechecked; this small training set is not a substitute for a current information source.

### Verification and evidence

Checks performed in this work: re-score **84 stored evaluation records** (21 questions × two phases × two settings); verify **8 Knowledge Question records**; compare the full data hashes and pre-training outputs; verify optimizer steps, actual training precision, source code, and independent kernels. All checks passed. The checks are recorded in [validation.json](results/validation.json) and can be rerun with `python protocol/validate_results.py` from this directory.

| Required submission evidence | 20 epochs | 10 epochs |
|---|---|---|
| Result/settings/training flag | [result.json](results/baseline_epochs20/result.json) | [result.json](results/comparison_epochs10/result.json) |
| Original before-training outputs | [eval_before.json](results/baseline_epochs20/eval_before.json) | [eval_before.json](results/comparison_epochs10/eval_before.json) |
| Original after-training outputs | [eval_after.json](results/baseline_epochs20/eval_after.json) | [eval_after.json](results/comparison_epochs10/eval_after.json) |
| Section 5 comparison | [CSV](results/baseline_epochs20/section5_comparison.csv) | [CSV](results/comparison_epochs10/section5_comparison.csv) |
| Remaining errors and raw examples | [JSON](results/baseline_epochs20/remaining_failures.json) | [JSON](results/comparison_epochs10/remaining_failures.json) |
| Section 6 full outputs | [JSON](results/baseline_epochs20/knowledge_questions.json) | [JSON](results/comparison_epochs10/knowledge_questions.json) |
| Loss, runtime, steps | [training_metrics.json](results/baseline_epochs20/training_metrics.json) | [training_metrics.json](results/comparison_epochs10/training_metrics.json) |
| Effective training arguments | [JSON](results/baseline_epochs20/training_arguments.json) | [JSON](results/comparison_epochs10/training_arguments.json) |

Reproduction notebooks: [20 epochs](notebooks/basic_epochs20.ipynb), [10 epochs](notebooks/basic_epochs10.ipynb). Actual CLI execution output: [10-epoch executed notebook](notebooks/basic_epochs10_output.ipynb). Source and control checks: [experiment protocol](protocol/experiment_protocol.json).

All required JSON files were exported from `/content/qlora_lab/`. The original 20-epoch directory was preserved before the comparison; after the comparison, `/content/qlora_lab/` contained the 10-epoch run. Full local backups include adapters and checkpoints for both settings; their checksums are in [local_backup_manifest.json](protocol/local_backup_manifest.json).
