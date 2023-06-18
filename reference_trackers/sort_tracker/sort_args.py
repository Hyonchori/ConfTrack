from types import SimpleNamespace


def make_sort_args():
    args = SimpleNamespace()

    args.track_thresh = 0.6
    args.vertical_thresh = 1.6
    args.min_box_area = 100

    return args
