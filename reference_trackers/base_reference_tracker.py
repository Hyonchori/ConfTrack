REFERENCE_TRACKERS = {
    0: 'SORT',
    1: 'DeepSORT',
    2: 'BYTE',
    3: 'OC-SORT',
    4: 'DeepOCSORT'
}


class ReferenceTracker:
    def __init__(self, select_tracker: int, target_select: str):
        self.select_tracker = select_tracker

        if select_tracker == 0:  # load SORT
            print('\nSORT trakcer is selected!')
            from .sort_tracker.sort_args import make_sort_args
            from .sort_tracker.sort import Sort
            self.tracker_args = make_sort_args()
            self.tracker_args.mot20 = target_select == 'MOT20'
            self.tracker = Sort(
                det_thresh=self.tracker_args.track_thresh
            )

        elif select_tracker == 1:  # load DeepSORT tracker
            print('\nDeepSORT trakcer is selected!')
            from .deepsort_tracker.deepsort_args import make_deepsort_args
            from .deepsort_tracker.deepsort import DeepSort
            self.tracker_args = make_deepsort_args()
            self.tracker_args.mot20 = target_select == 'MOT20'
            self.tracker = DeepSort(
                model_path=self.tracker_args.model_path,
                min_confidence=self.tracker_args.track_thresh
            )

        elif select_tracker == 2:  # load BYTETrack
            print('\nBYTE trakcer is selected!')
            from .byte_tracker.byte_args import make_byte_args
            from .byte_tracker.byte_tracker import BYTETracker
            self.tracker_args = make_byte_args()
            self.tracker_args.mot20 = target_select == 'MOT20'
            self.tracker = BYTETracker(self.tracker_args)

        elif select_tracker == 3:  # load OCSORT
            print('\nOC-SORT trakcer is selected!')
            from .ocsort_tracker.oc_args import make_oc_args
            from .ocsort_tracker.ocsort import OCSort
            self.tracker_args = make_oc_args()
            self.tracker_args.mot20 = target_select == 'MOT20'
            self.tracker = OCSort(
                det_thresh=self.tracker_args.track_thresh,
                iou_threshold=self.tracker_args.iou_thresh,
                asso_func=self.tracker_args.asso,
                delta_t=self.tracker_args.deltat,
                inertia=self.tracker_args.inertia
            )

        elif select_tracker == 4:  # load DeepOCSORT tracker
            print('\nDeepOCSORT tracker is selected!')
            from .deepocsort_tracker.deepoc_args import make_deepoc_args
            from .deepocsort_tracker.ocsort import OCSort
            self.tracker_args = make_deepoc_args()
            self.tracker_args.mot20 = target_select == 'MOT20'
            oc_sort_args = dict(
                args=self.tracker_args,
                det_thresh=self.tracker_args.track_thresh,
                iou_threshold=self.tracker_args.iou_thresh,
                asso_func=self.tracker_args.asso,
                delta_t=self.tracker_args.deltat,
                inertia=self.tracker_args.inertia,
                w_association_emb=self.tracker_args.w_assoc_emb,
                alpha_fixed_emb=self.tracker_args.alpha_fixed_emb,
                embedding_off=self.tracker_args.emb_off,
                cmc_off=self.tracker_args.cmc_off,
                aw_off=self.tracker_args.aw_off,
                aw_param=self.tracker_args.aw_param,
                new_kf_off=self.tracker_args.new_kf_off,
                grid_off=self.tracker_args.grid_off,
            )
            self.tracker = OCSort(**oc_sort_args)
        else:
            raise Exception(f'select_tracker should be one of {REFERENCE_TRACKERS}')

        self.ori_thresh = self.tracker_args.track_thresh if hasattr(self.tracker_args, 'track_thresh') else \
            self.tracker_args.track_high_thresh

    def initialize(self, video_name, target_select):
        self.tracker.initialize(self.tracker_args, video_name, target_select)

    def update(self, det, img_shape, img, img_idx, img_for_extractor=None):
        output = []

        # update OCSORT/SORT
        if self.select_tracker == 0 or self.select_tracker == 3:
            online_targets = self.tracker.update(det, img_shape, img_shape)
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > self.tracker_args.vertical_thresh
                if tlwh[2] * tlwh[3] > self.tracker_args.min_box_area and not vertical:
                    output.append([*t[:4], 1, tid])

        # update BYTE
        elif self.select_tracker == 2:
            online_targets = self.tracker.update(det, img_shape, img_shape)
            for t in online_targets:
                tlwh = t.tlwh
                conf = t.score
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.tracker_args.vertical_thresh
                if tlwh[2] * tlwh[3] > self.tracker_args.min_box_area and not vertical:
                    output.append([*t.tlbr, conf, tid])

        # update DeepSORT
        elif self.select_tracker == 1:
            online_targets = self.tracker.update(det, img_shape, img_shape, img)
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > self.tracker_args.vertical_thresh
                if tlwh[2] * tlwh[3] > self.tracker_args.min_box_area and not vertical:
                    output.append([*t[:4], 1, tid])

        # update DeepOCSORT
        elif self.select_tracker == 4:
            online_targets = self.tracker.update(det, img, img_for_extractor, img_idx)
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > self.tracker_args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.tracker_args.min_box_area and not vertical:
                    output.append([*t[:4], 1, tid])

        return output
