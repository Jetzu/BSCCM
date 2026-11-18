#!/usr/bin/env python3
"""Freeze hyperparameter choices from training-slide split-1 autofl only.
Call before held-out-slide evaluation. Does not read eval.json or clinical data.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',nargs='+',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); groups={}
    for run in a.runs:
        path=Path(run); v=json.loads((path/'validation.json').read_text())
        assert v['validation_only'] and not v['smoke']
        score=v['M1_autofl_truth_r']; assert score is not None
        key=(v['arm'],v['holdout']); groups.setdefault(key,[]).append((score,str(path.resolve())))
    result=[]
    for (arm,holdout),candidates in sorted(groups.items()):
        # Ties broken by path, not any other evaluation metric.
        score,run=sorted(candidates,key=lambda x:(-x[0],x[1]))[0]
        result.append(dict(arm=arm,holdout=holdout,selected_run=run,training_slide_fov1_autofl_r=score,candidates=candidates))
    out=Path(a.out)
    if out.exists():
        assert json.loads(out.read_text())==json.loads(json.dumps(result)),'selection is frozen; use a new experiment output'
    else:
        out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
