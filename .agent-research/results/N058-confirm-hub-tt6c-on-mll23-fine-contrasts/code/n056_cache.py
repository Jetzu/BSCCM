"""Single-file transactional inference cache; no per-batch files or concatenated copy."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import numpy as np

def blob(x):
    f=io.BytesIO();np.save(f,np.asarray(x),allow_pickle=False);return f.getvalue()

def array(b):return np.load(io.BytesIO(b),allow_pickle=False)

def read(path):
    with sqlite3.connect(path) as db:
        meta=json.loads(db.execute('SELECT value FROM meta').fetchone()[0])
        rows=db.execute('SELECT pred,z,zviews FROM batches ORDER BY start').fetchall()
        if sum(len(array(r[0])) for r in rows)!=len(meta['ids']):raise ValueError('incomplete cache')
        out={k:np.concatenate([array(r[i]) for r in rows]) for i,k in enumerate(['pred','z','zviews'])}
        out.update({k:np.asarray(v) for k,v in meta.items() if k!='bs'})
        return out

def predict(path,predictor,ids,load_image,bs,role,classes,check_stop):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Streaming hash avoids reading a large model checkpoint into memory.
    digest=hashlib.sha256()
    with open(predictor.ckpt,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    meta=dict(ids=np.asarray(ids).tolist(),names=predictor.names,role=role,model_sha=digest.hexdigest(),bs=bs)
    if classes is not None:meta['cls']=np.asarray(classes).tolist()
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('CREATE TABLE IF NOT EXISTS meta(value TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS batches(start INTEGER PRIMARY KEY,pred BLOB,z BLOB,zviews BLOB)')
        old=db.execute('SELECT value FROM meta').fetchone()
        if old and json.loads(old[0])!=meta:raise ValueError('stale inference cache contract')
        if not old:db.execute('INSERT INTO meta VALUES (?)',(json.dumps(meta),));db.commit()
        completed={r[0] for r in db.execute('SELECT start FROM batches')}
        for start in range(0,len(ids),bs):
            check_stop()
            if start in completed:continue
            p,z,v=predictor.predict([load_image(i) for i in range(start,min(start+bs,len(ids)))])
            with db:db.execute('INSERT INTO batches VALUES (?,?,?,?)',(start,blob(p),blob(z),blob(v)))
    return read(path)
