import logging
import os.path as osp

def compare(img_name):
    key = img_name.split('/')[-1].split('.')[0]
    return int(key)


def covis_from_index(img_lists, covis_pairs_out, num_matched, gap=3):
    """Get covis images by image id."""
    pairs = []
    img_lists.sort(key=compare)

    for i in range(len(img_lists)):
        count = 0
        j = i + 1
        
        while j < len(img_lists) and count < num_matched:
            if osp.dirname(img_lists[j]) == osp.dirname(img_lists[i]):
                index1 = int(osp.basename(img_lists[i]).split('.')[0])
                index2 = int(osp.basename(img_lists[j]).split('.')[0])
                if (index2 - index1) % gap == 0:
                    count += 1
                    pair = (img_lists[i], img_lists[j])
                    pairs.append(pair) 

            j += 1
    
    with open(covis_pairs_out, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
    
    logging.info('Finishing getting covis image pairs.')    