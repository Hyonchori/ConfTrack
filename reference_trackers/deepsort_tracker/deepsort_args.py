from pathlib import Path
from types import SimpleNamespace

FILE = Path(__file__)


def make_deepsort_args():
    args = SimpleNamespace()

    args.model_path = f'{FILE.parents[0]}/ckpt.t7'
    args.track_thresh = 0.6
    args.vertical_thresh = 1.6
    args.min_box_area = 100

    return args
