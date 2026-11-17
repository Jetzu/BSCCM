#!/usr/bin/env python3
"""Merge completed single-protocol/seed jobs without mixing LF teacher runs."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def write(path, rows):
    keys = sorted({k for r in rows for k in r})
    tmp = path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows,refs,seen=[],[],set()
    sources=[]
    fields=('lf_teacher','protocol','arm','family','metric','target','contrast','truth','split','smoke')
    for completion in sorted(a.root.rglob('completion.json')):
        folder=completion.parent
        if not json.loads(completion.read_text()).get('completed'):
            continue
        sources.append(str(folder))
        for r in json.loads((folder/'metrics.json').read_text()):
            key=tuple(str(r.get(k,'')) for k in (*fields,'seed'))
            if key in seen:
                # The independent raw PBC baseline is identical across jobs.
                if r['arm']=='raw_giemsa':
                    continue
                raise ValueError('duplicate run/metric: '+repr(key))
            seen.add(key);rows.append(r)
        refs.extend(json.loads((folder/'historical_references.json').read_text()))
    if not sources:
        raise ValueError('no completed runs found')
    unique_refs={json.dumps(r,sort_keys=True):r for r in refs}
    write(a.out/'metrics.csv',rows)
    write(a.out/'comparison.csv',rows+list(unique_refs.values()))
    groups={}
    for r in rows:
        v=r.get('value')
        if v is not None and math.isfinite(float(v)):
            key=tuple(str(r.get(k,'')) for k in fields)
            groups.setdefault(key,[]).append(float(v))
    summaries=[dict(zip(fields,k),n_alignment_seeds=len(v),mean=mean(v),sd=stdev(v) if len(v)>1 else '')
               for k,v in groups.items()]
    write(a.out/'summary_by_teacher.csv',summaries)
    # Average alignment seeds inside each LF teacher before averaging teachers.
    tf=fields[1:];tg={}
    for r in summaries:
        if not r['lf_teacher']:
            continue
        tg.setdefault(tuple(r[k] for k in tf),[]).append(r['mean'])
    ts=[dict(zip(tf,k),n_teachers=len(v),mean=mean(v),sd_between_teachers=stdev(v) if len(v)>1 else '',
             clinical_replication_status='>=3 teachers' if len(v)>=3 else 'exploratory: <3 LF teachers')
        for k,v in tg.items()]
    write(a.out/'summary_across_teachers.csv',ts)
    (a.out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(f'Merged {len(sources)} completed runs, {len(rows)} current metric rows')


if __name__=='__main__':
    main()
