from typing import List
import numpy as np

from .base_detection import BaseDetection


def get_detection(
        xyxy: np.ndarray,  # [x1, y1, x2, y2]
        conf: float,
        cls: int,
        feature: np.ndarray = None
):
    return BaseDetection(
        xyxy=xyxy,
        conf=conf,
        cls=cls,
        feature=feature
    )


def get_detections(
        cfg,
        predictions: np.ndarray,  # detector output: [x1, y1, x2, y2, confidence, class]
) -> List[BaseDetection]:
    detections = []

    for res in predictions:
        conf = float(res[4])
        if conf < cfg.detection_low_thr:
            continue
        bbox = res[:4]
        cls = int(res[5])
        detections.append(
            get_detection(xyxy=bbox, conf=conf, cls=cls)
        )

    return detections