from typing import List

import lap
import numpy as np

from ..track.base_track import BaseTrack, TrackState
from ..detection.base_detection import BaseDetection
from ..cost.cost_matrices import get_iou_cost, get_embedding_cost, get_confidence_fused_cost


def linear_assignment(cost_mat, row_indices, col_indices, matching_thresh: float = 0.8, gate_mat=None):
    if cost_mat.size == 0:
        return [], row_indices, col_indices
    matches = []
    if gate_mat is not None:
        cost_mat[gate_mat == 0] = np.inf
    cost, x, y = lap.lapjv(cost_mat, extend_cost=True, cost_limit=matching_thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.asarray(row_indices)[np.where(x < 0)[0]]
    unmatched_b = np.asarray(col_indices)[np.where(y < 0)[0]]
    matches = [[row_indices[pair[0]], col_indices[pair[1]]] for pair in matches]
    return matches, unmatched_a.tolist(), unmatched_b.tolist()


def associate_conftrack(
        cfg,
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        img_for_extractor: np.ndarray = None,
        extractor=None
):
    unmatched_trk_indices = list(range(len(trk_list)))
    unmatched_det_indices = list(range(len(det_list)))

    if len(det_list) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    det_high_indices = [i for i in unmatched_det_indices
                        if det_list[i].conf >= cfg.detection_high_thr]
    det_low_indices = [i for i in unmatched_det_indices
                       if cfg.detection_low_thr <= det_list[i].conf < cfg.detection_high_thr]
    det_xyxys = [det_list[i].xyxy for i in det_high_indices]

    if det_xyxys and extractor is not None:
        det_feats = extractor(det_xyxys, img_for_extractor)
        for i, det_feat in zip(det_high_indices, det_feats):
            det_list[i].feature = det_feat

    trk_conf_indices = [i for i in unmatched_trk_indices
                        if trk_list[i].is_lost() or trk_list[i].is_confirmed() or
                        (trk_list[i].is_tentative() and trk_list[i].conf >= 0.7)]
    trk_tent_indices = [i for i in unmatched_trk_indices
                        if trk_list[i].is_tentative() and trk_list[i].conf < 0.7]

    ''' First matching: high_confident_track <-> high_confident_detection (from BoTSORT) '''
    iou_cost_first, iou_gate_first = get_iou_cost(trk_list, det_list, trk_conf_indices, det_high_indices,
                                                  iou_dist_thresh=cfg.first_matching_iou_thresh)
    emb_cost, emb_gate = get_embedding_cost(trk_list, det_list, trk_conf_indices, det_high_indices,
                                            embedding_dist_thresh=cfg.first_matching_emb_thresh)

    if cfg.use_CFCM:
        iou_cost_first, _ = get_confidence_fused_cost(iou_cost_first, det_list, det_high_indices)
        emb_cost, _ = get_confidence_fused_cost(emb_cost, det_list, det_high_indices)

    emb_cost[emb_gate == 0] = 1.0
    emb_cost[iou_gate_first == 0] = 1.0

    cost_mat = np.minimum(iou_cost_first, emb_cost)  # cost matrix from BoTSORT

    matches_first, unmatched_trk_conf_indices, unmatched_det_high_indices = linear_assignment(
        cost_mat, trk_conf_indices, det_high_indices,
        matching_thresh=cfg.first_matching_thresh, gate_mat=iou_gate_first
    )

    ''' Second matching: unmatched_high_confident_track <-> low_confident_detection (from BYTETrack) '''
    r_trk_conf_indices = [i for i in unmatched_trk_conf_indices if
                          trk_list[i].is_confirmed() or trk_list[i].is_lost()]

    iou_cost_second, iou_gate_second = get_iou_cost(trk_list, det_list, r_trk_conf_indices, det_low_indices,
                                                    iou_dist_thresh=cfg.second_matching_iou_thresh)
    matches_second, unmatched_r_trk_conf_indices, unmatched_det_low_indices = linear_assignment(
        iou_cost_second, r_trk_conf_indices, det_low_indices,
        matching_thresh=cfg.second_matching_iou_thresh, gate_mat=iou_gate_second
    )

    if cfg.use_LCTM:
        ''' Low confidence matching: low_confident_track <-> unmatched_high_confident_detection (LCTM) '''
        iou_cost_tent_conf, _ = get_iou_cost(trk_list, trk_list,
                                             trk_tent_indices, unmatched_r_trk_conf_indices)
        matches_tent_conf, unmatched_trk_tent_indices, unmatched_r_trk_conf_indices = linear_assignment(
            iou_cost_tent_conf, trk_tent_indices, unmatched_r_trk_conf_indices,
            matching_thresh=cfg.low_confidence_matching_negli_thresh, gate_mat=None
        )

        iou_cost_tent_high, _ = get_iou_cost(trk_list, det_list,
                                             unmatched_trk_tent_indices, unmatched_det_high_indices)
        matches_tent_high, _, _ = linear_assignment(
            iou_cost_tent_high, unmatched_trk_tent_indices, unmatched_det_high_indices,
            matching_thresh=cfg.low_confidence_matching_thresh, gate_mat=None
        )
    else:
        matches_tent_high = []

    matches = matches_first + matches_second + matches_tent_high
    unmatched_trk_indices = list(set(unmatched_trk_indices) - set([x[0] for x in matches]))
    unmatched_det_indices = list(set(unmatched_det_indices) - set([x[1] for x in matches]))

    return matches, unmatched_trk_indices, unmatched_det_indices
