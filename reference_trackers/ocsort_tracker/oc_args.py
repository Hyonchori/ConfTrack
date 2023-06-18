from types import SimpleNamespace


def make_oc_args():
    args = SimpleNamespace()

    args.track_thresh = 0.6
    args.iou_thresh = 0.3
    args.min_hits = 3
    args.inertia = 0.2
    args.deltat = 3
    args.track_buffer = 30
    args.match_thresh = 0.9
    args.vertical_thresh = 1.6
    args.min_box_area = 10
    args.mot20 = False
    args.asso = 'iou'

    return args
