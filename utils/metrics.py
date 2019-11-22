import numpy as np
from cremi.evaluation.rand import adapted_rand
from cremi.evaluation.voi import voi
from cremi.evaluation.border_mask import create_border_mask_2d


def compute_metrics(labels, gt):
    # mask boundary pixels as label 0
    gt += 1  # label zero will be for background and is automatically ignored by adapted rand
    for batch_id in range(gt.shape[0]):
        mask = create_border_mask_2d(gt[batch_id], max_dist=0)
        gt[batch_id][mask] = 0

    arand_list = []
    voi_list = []
    for batch_id in range(gt.shape[0]):
        gt_slice = gt[batch_id].ravel()
        lables_slice = labels[batch_id].ravel()

        arand_slice = adapted_rand(lables_slice, gt_slice)  # ignores ground truth label 0 automatically
        voi_score_slice = np.array(voi(lables_slice, gt_slice, ignore_groundtruth=[0]))

        arand_list.append(arand_slice)
        voi_list.append(voi_score_slice)

    arand = np.mean(np.array(arand_list), axis=0)
    voi_score = np.mean(np.array(voi_list), axis=0)
    CREMI_score = np.sqrt(voi_score.sum() * arand)

    return {"arand": arand,
            "voi": voi_score,
            "CREMI_score": CREMI_score,
            }


