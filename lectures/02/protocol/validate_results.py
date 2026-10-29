"""Recheck the stored records using the course's published scoring rules."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def classify_format(text):
    try: obj=json.loads(text)
    except (ValueError,TypeError):return False,None
    valid=(isinstance(obj,dict) and set(obj)=={'answer','needs_review'}
           and isinstance(obj.get('answer'),str) and type(obj.get('needs_review')) is bool)
    return valid,obj


def score(text,gold):
    valid,obj=classify_format(text)
    if not valid:
        match=re.search(r'\{.*\}',text,re.S)
        try:obj=json.loads(match.group()) if match else None
        except (ValueError,TypeError):obj=None
    content=(isinstance(obj,dict) and obj.get('needs_review')==gold['needs_review']
             and all(word in obj.get('answer','') for word in gold['must_include']))
    return valid,bool(content),obj


def validate():
    findings={};runs={};checked=0
    for label,epochs in [('baseline_epochs20',20),('comparison_epochs10',10)]:
        path=ROOT/'results'/label
        result=json.loads((path/'result.json').read_text())
        gold=json.loads((path/'gold.json').read_text())
        train=[json.loads(line) for line in (path/'train.jsonl').read_text().splitlines()]
        test=[json.loads(line) for line in (path/'test.jsonl').read_text().splitlines()]
        assert len(train)==14 and len(test)==len(gold)==21
        assert not ({x['question'] for x in train}&{x['question'] for x in test})
        assert result['config']['num_train_epochs']==epochs
        assert result['executed_training_myself'] is True
        evals={}
        for phase in ['before','after']:
            data=json.loads((path/f'eval_{phase}.json').read_text())
            rows=data['rows'];assert len(rows)==21
            assert [row['question'] for row in rows]==[row['question'] for row in test]
            for row in rows:
                f,c,_=score(row['raw'],gold[row['question']])
                assert f==row['format_ok'] and c==row['content_ok']
                assert (f and c)==row['both_ok']
                assert row['gold_needs_review']==gold[row['question']]['needs_review']
                checked+=1
            for field in ['format','content','both']:
                actual=sum(row[field+'_ok'] for row in rows)/21
                assert abs(actual-data['summary'][field+'_rate'])<1e-12
            assert data['summary']==result[phase]
            evals[phase]=data
        regression=[b['question'] for b,a in zip(evals['before']['rows'],evals['after']['rows']) if b['both_ok'] and not a['both_ok']]
        assert regression==result['regressed']
        failures=[]
        for before,after in zip(evals['before']['rows'],evals['after']['rows']):
            if after['both_ok']:continue
            valid,_,obj=score(after['raw'],gold[after['question']]);types=[]
            if not valid:types.append('broken_format')
            if isinstance(obj,dict):
                if gold[after['question']]['needs_review'] and obj.get('needs_review') is False:types.append('assertion_without_evidence')
                elif not gold[after['question']]['needs_review'] and obj.get('needs_review') is True:types.append('evasion_despite_evidence')
                elif not after['content_ok']:types.append('required_keyword_mismatch')
            failures.append({'question':after['question'],'expected_needs_review':gold[after['question']]['needs_review'],
                             'types':types,'before_output':before['raw'],'after_output':after['raw']})
        for row in result['knowledge']:
            for phase in ['Before','After']:
                f,obj=classify_format(row[phase+'-training output'])
                expected='format_error' if not f else ('needs_review' if obj['needs_review'] else 'asserted')
                assert expected==row[phase+'-training classification']
        assert len(result['knowledge'])==4
        metrics=json.loads((path/'training_metrics.json').read_text())
        args=json.loads((path/'training_arguments.json').read_text())
        assert metrics['global_step']==epochs*2 and metrics['epoch']==epochs
        for name,digest in metrics['data_hashes'].items():assert hashlib.sha256((path/name).read_bytes()).hexdigest()==digest
        (path/'remaining_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
        findings[label]={'epochs':epochs,'before':result['before'],'after':result['after'],'regressed':regression,
                         'failure_count':len(failures),'failures':failures,'training':{k:v for k,v in metrics.items() if k!='log_history'}}
        runs[label]={'result':result,'metrics':metrics,'arguments':args,'evals':evals}
    a,b=runs.values()
    assert [k for k in a['result']['config'] if a['result']['config'][k]!=b['result']['config'][k]]==['num_train_epochs']
    assert [k for k in a['arguments'] if a['arguments'][k]!=b['arguments'][k]]==['num_train_epochs']
    assert a['result']['versions']==b['result']['versions']
    assert a['metrics']['versions']==b['metrics']['versions']
    assert a['metrics']['data_hashes']==b['metrics']['data_hashes']
    assert a['metrics']['hardware']==b['metrics']['hardware']
    assert a['metrics']['kernel_pid']!=b['metrics']['kernel_pid']
    assert a['evals']['before']['rows']==b['evals']['before']['rows']
    validation={'passed':True,'stored_eval_rows_rechecked':checked,'knowledge_rows_rechecked':8,
                'changed_parameter':'num_train_epochs','identical_pretraining_outputs':True,
                'identical_data_hashes':True,'fresh_comparison_kernel':True,
                'identical_other_effective_training_arguments':True,'runs':findings}
    (ROOT/'results/validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(validation,ensure_ascii=False,indent=2))
    return validation


if __name__=='__main__':validate()
