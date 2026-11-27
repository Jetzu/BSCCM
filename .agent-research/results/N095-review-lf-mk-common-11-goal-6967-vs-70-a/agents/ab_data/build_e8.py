import numpy as np, sys
sys.argv=['x']
W='/hnvme/workspace/v181bb10-bsccm'
src=open('/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N085-stronger-members-convnext-s-with-120-ep/n085_score.py').read()
# take definitions only (up to SETS)
defs=src.split("SETS = {")[0]
exec(defs)
keys=['n048_v9_base','n048_v9_zoom2','n053_cytoland','comp_base','comp_z2','comp_cyto','cnxs_plain','cnxs_comp']
for d in ('b10','b01'):
    m,G=members(d); P=np.mean([m[k] for k in keys],0)
    np.savez(f'/tmp/n095_ab/e8_{d}.npz',P=P,G=G,**{'m_'+k:m[k] for k in keys})
    print(d, score(P,G))
