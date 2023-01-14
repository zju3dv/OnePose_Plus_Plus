import numpy as np
from itertools import combinations

def exhaustive_all_pairs(img_list, covis_pairs_out):
    pair_ids = list(combinations(range(len(img_list)), 2))
    img_pairs = []
    for pair_id in pair_ids:
        img_pairs.append((img_list[pair_id[0]], img_list[pair_id[1]]))

    with open(covis_pairs_out, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in img_pairs))