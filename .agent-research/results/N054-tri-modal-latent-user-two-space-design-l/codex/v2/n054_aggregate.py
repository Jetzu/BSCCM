#!/usr/bin/env python3
"""Build the primary table strictly from frozen source-validation selections.
The secondary table retains every predeclared configuration and diagnostics.
No model fitting, selection, or test-based ranking occurs here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def read(path): return json.loads(Path(path).read_text())

def fingerprint(path):
    p=Path(path).resolve(); st=p.stat()
    return dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns)

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--units',nargs='+',required=True); p.add_argument('--lf-unit',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(argv); dest=Path(a.out); dest.mkdir(parents=True,exist_ok=True)
    code=hashlib.sha256(Path(__file__).with_name('n054_trimodal.py').read_bytes()).hexdigest()
    primary=[]; secondary=[]; detailed_primary=[]; detailed_secondary=[]; diagnostics=[]; selections=[]
    for unit in map(Path,a.units):
        completed=read(unit/'unit.done.json'); sel=read(unit/'selection.json')
        assert completed['code_signature']==sel['code_signature']==code,'unit/code signature mismatch'
        assert completed['selection']==fingerprint(unit/'selection.json'),'selection changed since completion'
        for family,item in sel['selected'].items():
            assert item['arm']==family
            assert item['validation']==fingerprint(Path(item['run'])/'validation.json'),'selected validation changed'
        selected_runs={item['run'] for item in sel['selected'].values()}
        selections.append(dict(path=str((unit/'selection.json').resolve()),sha256=hashlib.sha256((unit/'selection.json').read_bytes()).hexdigest(),selected=sel['selected']))
        grid={r['run']:r for r in sel['grid']}
        for run in sorted(unit.iterdir()):
            if not run.is_dir() or not (run/'eval.json').exists(): continue
            ev=read(run/'eval.json'); done=read(run/'eval.done.json'); args=done['args']
            assert done['code_signature']==code
            assert all(fingerprint(o['path'])==o for o in done['outputs']),'evaluation artifacts changed'
            path=str(run.resolve()); chosen=ev['arm']=='A0' or path in selected_runs
            if ev['arm']!='A0':
                assert path in grid,'unregistered test configuration'
                assert grid[path]['validation']==fingerprint(run/'validation.json')
            row=dict(holdout=ev['holdout'],arm=ev['arm'],sigma_bridge=args['sigma_bridge'] if ev['arm'] not in ('A0','A3','direct') else None,
                w_sig=args['w_sig'] if ev['arm']=='A3' else None,run=path,primary=chosen,
                validation_autofl_r=grid[path]['r'] if path in grid else None,M1_autofl_truth_r=ev['M1_autofl_truth_r'],
                n_gs=ev['M3']['n'],R_at_1=ev['M3']['R_at_1'],R_at_10=ev['M3']['R_at_10'],
                linear_cka=ev['M4_GS_LF']['linear_cka'],mutual_knn_overlap=ev['M4_GS_LF']['mutual_knn_overlap'])
            row.update(ev.get('modality_gap',{}))
            clinical=run/'clinical/clinical.json'
            detailed=[dict(holdout=ev['holdout'],arm=ev['arm'],run=path,**r) for r in ev['per_target']]
            if chosen:
                assert clinical.exists(),f'missing selected clinical output: {run}'
                detailed += [dict(holdout=ev['holdout'],arm=ev['arm'],run=path,metric='M6',**r) for r in read(clinical)['contrasts']]
            secondary.append(row); detailed_secondary.extend(detailed)
            if chosen: primary.append(row); detailed_primary.extend(detailed)
            if 'oracle_A1_LF' in ev:
                diagnostics.append(dict(holdout=ev['holdout'],arm='oracle_A1-LF',sigma_bridge=args['sigma_bridge'],run=path,**ev['oracle_A1_LF']))
            if ev['arm']=='direct': diagnostics.append(row)
        assert len([r for r in primary if r['holdout']==sel['holdout']])==5,'expected A0 and four selected families'
    write_csv(dest/'primary.csv',primary); write_csv(dest/'secondary_full_grid.csv',secondary)
    write_csv(dest/'primary_metrics.csv',detailed_primary); write_csv(dest/'secondary_metrics.csv',detailed_secondary)
    write_csv(dest/'diagnostics.csv',diagnostics)
    completion=Path(a.lf_unit)/'completion/completion'
    for name in ('completion.csv','completion.json','coherence.csv'): shutil.copy2(completion/name,dest/name)
    metadata=dict(code_signature=code,aggregate_signature=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_selections=selections,n_primary=len(primary),n_grid=len(secondary),n_diagnostics=len(diagnostics),
        completion_source=fingerprint(completion/'completion.json'))
    (dest/'aggregation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(primary=len(primary),full_grid=len(secondary),diagnostics=len(diagnostics),out=str(dest))))

if __name__=='__main__': main()
