"""Contrast dictionary assignments copied verbatim from apply_gs_student.py."""
L, N = ['lymphocyte'], ['neutrophil_segmented']
C = {'CD16 neutrophil>lymphocyte': ('single:CD16', ['neutrophil'], ['lymphocyte']), 'CD16 neutrophil>eosinophil': ('single:CD16', ['neutrophil'], ['eosinophil']),
     'CD14 monocyte>neutrophil+lymphocyte': ('single:CD14', ['monocyte'], ['neutrophil', 'lymphocyte']),
     'grpCD3/19/56 lymphocyte>neutrophil+monocyte': ('grp:CD3/CD19/CD56', ['lymphocyte'], ['neutrophil', 'monocyte']),
     'CD3 lymphocyte>neutrophil+monocyte': ('single:CD3', ['lymphocyte'], ['neutrophil', 'monocyte']), 'CD45 lymphocyte>neutrophil': ('single:CD45', ['lymphocyte'], ['neutrophil']),
     'CD123 basophil>rest': ('single:CD123', ['basophil'], ['neutrophil', 'lymphocyte', 'monocyte', 'eosinophil']), 'HLA-DR monocyte>neutrophil': ('single:HLA-DR', ['monocyte'], ['neutrophil']),
     # added 2026-11-09 BEFORE seeing results (held-out classes, never used for alignment): CD16 rises with granulocyte maturation;
     # erythroblasts are CD45-dim/negative
     'CD16 neutrophil>ig (held-out)': ('single:CD16', ['neutrophil'], ['ig']), 'CD45 lymphocyte>erythroblast (held-out)': ('single:CD45', ['lymphocyte'], ['erythroblast'])}
MC = {'LGL>lym CD56': ('single:CD56', ['lymphocyte_large_granular'], L, 1), 'LGL>lym CD16': ('single:CD16', ['lymphocyte_large_granular'], L, 1),
          'hairy>lym CD123': ('single:CD123', ['hairy_cell'], L, 1), 'hairy>lym CD19': ('single:CD19', ['hairy_cell'], L, 1),
          'hairy>lym HLA-DR': ('single:HLA-DR', ['hairy_cell'], L, 1), 'hairy<lym CD3': ('single:CD3', ['hairy_cell'], L, -1),
          'plasma<lym CD45': ('single:CD45', ['plasma_cell'], L, -1), 'eos<segneut CD16': ('single:CD16', ['eosinophil'], N, -1),
          'band<segneut CD16': ('single:CD16', ['neutrophil_band'], N, -1), 'myelocyte<segneut CD16': ('single:CD16', ['myelocyte', 'metamyelocyte'], N, -1),
          'baso>rest CD123': ('single:CD123', ['basophil'], ['lymphocyte', 'neutrophil_segmented', 'monocyte', 'eosinophil'], 1),
          'normoblast<lym CD45': ('single:CD45', ['normoblast'], L, -1),
          # lineage references
          'segneut>lym CD16': ('single:CD16', N, L, 1), 'mono>lym+neut CD14': ('single:CD14', ['monocyte'], L + N, 1), 'lym>neut CD3': ('single:CD3', L, N, 1)}

PRIMARY_PBC = ['CD16 neutrophil>eosinophil', 'CD123 basophil>rest',
               'CD16 neutrophil>ig (held-out)', 'CD45 lymphocyte>erythroblast (held-out)']
PRIMARY_MLL = [k for k in MC if k not in ['segneut>lym CD16','mono>lym+neut CD14','lym>neut CD3']]
def primary(domain,key):return key in (PRIMARY_PBC if domain=='PBC' else PRIMARY_MLL)
