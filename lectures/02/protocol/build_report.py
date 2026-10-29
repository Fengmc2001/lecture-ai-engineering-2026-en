"""Build the course report from the validated saved artifacts."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load(label,name):return json.loads((ROOT/'results'/label/name).read_text())

def pct(value):return f'{value*100:.2f}% ({round(value*21)}/21)'

def build():
    validation=json.loads((ROOT/'results/validation.json').read_text());assert validation['passed']
    labels=['baseline_epochs20','comparison_epochs10']
    results=[load(label,'result.json') for label in labels]
    metrics=[load(label,'training_metrics.json') for label in labels]
    args=[load(label,'training_arguments.json') for label in labels]
    rows=[['Training epochs',20,10],['Optimizer steps',40,20],
          ['Before: format compliance',pct(results[0]['before']['format_rate']),pct(results[1]['before']['format_rate'])],
          ['Before: content accuracy',pct(results[0]['before']['content_rate']),pct(results[1]['before']['content_rate'])],
          ['After: format compliance',pct(results[0]['after']['format_rate']),pct(results[1]['after']['format_rate'])],
          ['After: content accuracy',pct(results[0]['after']['content_rate']),pct(results[1]['after']['content_rate'])],
          ['After: both format and content',pct(results[0]['after']['both_rate']),pct(results[1]['after']['both_rate'])],
          ['Last logged loss',f"{metrics[0]['final_logged_loss']:.6f}",f"{metrics[1]['final_logged_loss']:.6f}"],
          ['Mean loss across training',f"{metrics[0]['train_loss_mean']:.6f}",f"{metrics[1]['train_loss_mean']:.6f}"],
          ['Trainer runtime (seconds)',f"{metrics[0]['train_runtime_seconds']:.4f}",f"{metrics[1]['train_runtime_seconds']:.4f}"],
          ['Regressed questions (course definition)',len(results[0]['regressed']),len(results[1]['regressed'])]]
    with (ROOT/'results/basic_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Metric','Baseline: 20 epochs','Comparison: 10 epochs']);w.writerows(rows)
    table='\n'.join(['| Metric | Baseline: 20 epochs | Comparison: 10 epochs |','|---|---:|---:|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
    failures={label:load(label,'remaining_failures.json') for label in labels}
    explanations={
        'Can I submit a format other than PDF?':'The FAQ specifies a PDF, but the response asks for confirmation and sets needs_review=true. This is evasion despite available evidence.',
        'What happens if I miss the submission deadline?':'The FAQ states the 48-hour late-submission policy, but the response asks for confirmation. This is evasion despite available evidence.',
        'Do office hours require an appointment?':'The FAQ provides office-hour times but no appointment policy. The response repeats the times and sets needs_review=false instead of requesting confirmation.'}
    failure_table=['| Question | 20 epochs | 10 epochs | Interpretation |','|---|---|---|---|']
    for question in explanations:
        present=[any(x['question']==question for x in failures[label]) for label in labels]
        if any(present):failure_table.append('| '+question+' | '+' | '.join('Fail' if x else 'Pass' for x in present)+' | '+explanations[question]+' |')
    knowledge_table=['| Question | In FAQ | Before training (both runs) | After 20 epochs | After 10 epochs |','|---|---|---|---|---|']
    for a,b in zip(results[0]['knowledge'],results[1]['knowledge']):
        assert a['Question']==b['Question']
        assert a['Before-training classification']==b['Before-training classification']
        knowledge_table.append('| '+' | '.join([a['Question'],'Yes' if a['In FAQ'] else 'No',a['Before-training classification'],a['After-training classification'],b['After-training classification']])+' |')
    for label in labels:
        path=ROOT/'results'/label
        before=load(label,'eval_before.json');after=load(label,'eval_after.json')
        if not (path/'section5_comparison.csv').exists():
            with (path/'section5_comparison.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['label','n','format_rate','content_rate','both_rate','seconds']);w.writeheader();w.writerows([before['summary'],after['summary']])
        if not (path/'section5_per_question.csv').exists():
            with (path/'section5_per_question.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['question','expected_needs_review','before_output','before_format','before_content','after_output','after_format','after_content'])
                for b,a in zip(before['rows'],after['rows']):w.writerow([b['question'],b['gold_needs_review'],b['raw'],b['format_ok'],b['content_ok'],a['raw'],a['format_ok'],a['content_ok']])
    saved_seconds=metrics[0]['train_runtime_seconds']-metrics[1]['train_runtime_seconds']
    saved_pct=saved_seconds/metrics[0]['train_runtime_seconds']*100
    report=f'''# Lecture 02 Answers

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

{table}

The 20-epoch columns describe the original exercise. The 10-epoch columns are the Basic homework comparison. `Last logged loss` is the final 10-step logging-window average (step 40 or step 20), not an instantaneous final-minibatch loss. `Mean loss across training` is the trainer's separate `train_loss` metric. Runtime is the trainer-reported training runtime; model loading and evaluation are excluded.

Both runs have `regressed=[]` under the course definition: a question passing both format and content before training and failing afterward. This does not imply that every response improved. For example, the PDF-alternative question had a substantively relevant explanation before training but an incorrect review flag; after training it unnecessarily asked for confirmation. Both outputs fail the course content check.

### Remaining failures and their types

{chr(10).join(failure_table)}

After training there were **no format failures**. The 20-epoch run had one grounded-question evasion and one unsupported response to an ungrounded question. The 10-epoch run had two grounded-question evasions and the same unsupported appointment-policy response.

Full before/after text and expected review flags are preserved in the two `eval_*.json` files and `section5_per_question.csv`. The appointment response does not explicitly establish an appointment rule: it gives the schedule while marking `needs_review=false`, which is insufficient for this question.

### Section 6: Knowledge Questions

{chr(10).join(knowledge_table)}

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

The 10-epoch run used **{saved_seconds:.4f} fewer training seconds ({saved_pct:.1f}% less)** while retaining 21/21 format compliance. It answered one fewer test question correctly than the 20-epoch run: 18/21 versus 19/21, a difference of 4.76 percentage points. The additional error concerned the late-submission policy.

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
'''
    (ROOT/'answer.md').write_text(report)
    draft='Compared the default 20 epochs with a fresh-kernel 10-epoch run. Format accuracy reached 21/21 in both; content accuracy was 19/21 vs 18/21. Kept all required JSON, knowledge-question outputs and failure analysis. Rechecked 84 stored evaluation records and verified identical data and other settings. CLI execution and reporting used AI assistance.'
    assert len(draft)<=400
    (ROOT/'submission.md').write_text('# Omnicampus submission draft\n\nQ1: Basic\n\nQ2: https://github.com/Fengmc2001/lecture-ai-engineering-2026-en/tree/lec02\n\nQ3 ('+str(len(draft))+' characters):\n\n'+draft+'\n')
    (ROOT/'results/analysis_summary.json').write_text(json.dumps({'comparison_rows':rows,'training_seconds_saved':saved_seconds,'training_runtime_reduction_percent':saved_pct,'after_content_difference_count':1,'one_run_per_setting':True,'knowledge_classifications_changed':False},indent=2)+'\n')
    print(json.dumps({'answer_words':len(report.split()),'submission_characters':len(draft),'baseline_content':results[0]['after']['content_rate'],'comparison_content':results[1]['after']['content_rate']}))


if __name__=='__main__':build()
