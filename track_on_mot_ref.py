# Make tracking prediction on MOT17/MOT20 dataset using reference trackers

import argparse
import os
import sys
import time
import importlib
import warnings
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm

from custom_utils.torch_utils import select_device
from custom_utils.general_utils import increment_path, xywh2xyxy
from custom_utils.plot_utils import letterbox, plot_info, plot_detection, plot_bboxes

from datasets.MOT.mot_dataset import get_mot_videos, MOT_CLASSES

from detector.config_detector import get_detector
from detector.detector_utils import scale_coords, clip_coords

from tracker.detection.config_detection import get_detections
from reference_trackers.base_reference_tracker import ReferenceTracker

REFERENCE_TRACKERS = {
    0: 'SORT',
    1: 'DeepSORT',
    2: 'BYTE',
    3: 'OC-SORT',
    4: 'DeepOCSORT'
}


@torch.no_grad()
def main(args):
    # Arguments for MOT dataset
    mot_root = args.mot_root
    target_select = args.target_select
    target_split = args.target_split
    target_vid = args.target_vid
    if target_vid is not None and not isinstance(target_vid, list):
        target_vid = [target_vid]
    target_det = args.target_det
    if target_det is not None and not isinstance(target_det, list):
        target_det = [target_det]

    # Arguments for detector configuration
    trk_cfg_file = args.det_cfg_file
    sys.path.append((os.path.join(os.path.dirname(FILE), 'cfgs')))
    trk_cfg_file = importlib.import_module(trk_cfg_file)
    trk_cfg = trk_cfg_file.TrackerCFG()

    # Arguments for reference tracker
    select_tracker = args.select_tracker

    # General arguments for inference
    device = select_device(args.device)
    vis_progress_bar = args.vis_progress_bar
    run_name = args.run_name
    vis_det = args.vis_det
    vis_trk = args.vis_trk
    visualize = args.visualize
    view_size = args.view_size
    save_pred = args.save_pred
    save_vid = args.save_vid
    save_vid_fps = args.save_vid_fps

    # load detector using configuration
    detector = get_detector(
        cfg=trk_cfg,
        device=device
    )

    # load reference tracker
    reference_tracker = ReferenceTracker(
        select_tracker=select_tracker,
        target_select=target_select
    )

    # load MOT dataset
    mot_videos, remain_dets = get_mot_videos(
        mot_root=mot_root,
        target_select=target_select,
        target_split=target_split,
        target_vid=target_vid,
        target_det=target_det,
        cfg=trk_cfg,
        input_size=detector.input_size if detector.model is not None else view_size
    )

    # make save directory
    out_dir = f'{FILE.parents[0]}/results/reference_tracker/' \
              f'{REFERENCE_TRACKERS[select_tracker]}/{target_select}_{target_split}'
    save_dir = increment_path(Path(out_dir) / run_name, exist_ok=False)
    if save_pred or save_vid:
        save_dir.mkdir(parents=True, exist_ok=True)
        trk_cfg.save_opt(save_dir)
        print(f"\nSave directory '{save_dir}' is created!")

    # pre-inference
    if device.type != "cpu":
        if detector.model is not None:
            detector(torch.zeros(1, 3, *detector.input_size).to(device).
                     type_as(next(detector.model.parameters())))

    # iterate video
    start_time = time.time()
    for vid_idx, mot_dataset in enumerate(mot_videos):
        vid_name = mot_dataset.dataset.dataset_name
        print(f"\n--- Processing {vid_idx + 1} / {len(mot_videos)}'s video: {vid_name}")

        # initialize tracker's track list and track id
        reference_tracker.initialize(vid_name, target_split)

        # create iterator on current video
        time.sleep(0.5)
        iterator = tqdm(enumerate(mot_dataset), total=len(mot_dataset),
                        desc=f'{vid_idx + 1}/{len(mot_videos)}: {vid_name}') \
            if vis_progress_bar else enumerate(mot_dataset)

        if save_vid:
            save_vid_path = os.path.join(save_dir, f'{vid_name}.mp4')
            vid_writer = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         save_vid_fps, (view_size[1], view_size[0]))

        mot_trk_pred = ''
        ts_load = time.time()
        for i, iter_data in iterator:
            if mot_dataset.dataset.use_detector and mot_dataset.dataset.use_extractor:
                img_raw, img, img_ori_size, det = iter_data

            elif mot_dataset.dataset.use_detector and not mot_dataset.dataset.use_extractor:
                img_raw, img, det = iter_data

            elif not mot_dataset.dataset.use_detector and mot_dataset.dataset.use_extractor:
                img_raw, img_ori_size, det = iter_data

            else:
                img_raw, det = iter_data
            img_v = img_raw[0]
            te_load = time.time()

            if not vis_progress_bar:
                print(f'\n--- {vid_name}: {i + 1} / {len(mot_dataset)}')

            # make detection
            ts_det = time.time()
            if mot_dataset.dataset.use_detector:
                img = img.to(device).type_as(next(detector.model.parameters()))
                det = detector(img)[0]
                # print(det)
                # print(det.shape, ' private')
                if det is not None:
                    det = det.cpu().numpy()
                    scale_coords(detector.input_size, det, img_raw[0].shape[:2], center_pad=False)
                    if target_select == 'MOT20':
                        clip_coords(det, img_raw[0].shape[:2])
                else:
                    det = np.empty((0, 6))
            else:
                det = det[0]
                # print(det)
                # print(det.shape, ' byte')
                if len(det) != 0:
                    det = xywh2xyxy(det)
                else:
                    det = np.empty((0, 6))
            detections = get_detections(trk_cfg, det)
            te_det = time.time()

            # make tracking prediction
            ts_trk = time.time()
            if mot_dataset.dataset.use_extractor:
                trk_pred = reference_tracker.update(
                    det=det,
                    img_shape=img_raw[0].shape[:2],
                    img=img_raw[0],
                    img_idx=i,
                    img_for_extractor=img_ori_size
                )
            else:
                trk_pred = reference_tracker.update(
                    det=det,
                    img_shape=img_raw[0].shape[:2],
                    img=img_raw[0],
                    img_idx=i,
                )
            te_trk = time.time()

            # write tracking results
            if save_pred:
                for p in trk_pred:
                    trk_xyxy = p[:4]
                    trk_id = p[5]
                    trk_width = trk_xyxy[2] - trk_xyxy[0]
                    trk_height = trk_xyxy[3] - trk_xyxy[1]
                    mot_trk_pred += f'{i + 1},{trk_id},' + \
                                    f'{trk_xyxy[0]},{trk_xyxy[1]},{trk_width},{trk_height},-1,-1,-1,-1\n'

            # visualize detection and tracking
            ts_vis = time.time()
            if visualize:
                plot_info(img_v, f'{vid_name}: {i + 1} / {len(mot_dataset)}',
                          font_size=2, font_thickness=2)

                if vis_det:
                    plot_detection(img_v, detections, MOT_CLASSES, hide_cls=True, hide_confidence=False)

                if vis_trk:
                    plot_bboxes(img_v, trk_pred, hide_confidence=True)

                if view_size is not None:
                    img_v = letterbox(img_v, view_size, auto=False)[0]

                cv2.imshow(vid_name, img_v)
                keyboard_input = cv2.waitKey(0) & 0xff
                if keyboard_input == ord('q'):
                    break
                elif keyboard_input == 27:  # 27: esc
                    sys.exit()
            if save_vid:
                vid_writer.write(img_v)
            te_vis = time.time()

            if not vis_progress_bar:
                load_time = te_load - ts_load
                det_time = te_det - ts_det
                trk_time = te_trk - ts_trk
                vis_time = te_vis - ts_vis
                iter_time = te_vis - te_load
                print(f'load_time: {load_time:.4f}')
                print(f'det_time: {det_time:.4f}')
                print(f'trk_time: {trk_time:.4f}')
                print(f'vis_time: {vis_time:.4f}')
                print(f'\titer_total_time: {iter_time:.4f}')
            ts_load = time.time()

        if visualize:
            cv2.destroyWindow(vid_name)

        # save tracking prediction for TrackEval
        if save_pred:
            track_save_dir = os.path.join(save_dir, f'{target_select}-{target_split}', trk_cfg.tracker_name,
                                          'data')  # for using TrackEval code
            if not os.path.isdir(track_save_dir):
                os.makedirs(track_save_dir)

            pred_path = os.path.join(track_save_dir, f'{vid_name}.txt')
            with open(pred_path, 'w') as f:
                f.write(mot_trk_pred)
                # print(f'\ttrack prediction result is saved in "{pred_path}"!')

            if target_select == 'MOT17':
                for remain_det in remain_dets:
                    # for additional prediction file for TrackEval when using specific detection
                    remain_vid_name = '-'.join(vid_name.split('-')[:-1] + [remain_det])
                    remain_path = os.path.join(track_save_dir, f'{remain_vid_name}.txt')
                    with open(remain_path, 'w') as f:
                        f.write(mot_trk_pred)
                        # print(f'\ttrack prediction result is saved in "{remain_path}"!')
            time.sleep(0.05)

    if save_pred:
        print(f'\ntrack prediction results are saved in "{save_dir}"!')

    end_time = time.time()
    print(f'\nTotal elapsed time: {end_time - start_time:.2f}')


def get_args():
    parser = argparse.ArgumentParser()

    # Arguments for MOT17/MOT20 dataset
    mot_root = '/home/jhc/Desktop/dataset/open_dataset/MOT'  # path to MOT dataset
    parser.add_argument('--mot_root', type=str, default=mot_root)

    target_select = 'MOT17'  # select in ['MOT17', 'MOT20']
    parser.add_argument('--target_select', type=str, default=target_select)

    target_split = 'val'  # select in ['train', 'val', 'test']
    parser.add_argument('--target_split', type=str, default=target_split)

    target_vid = None  # None: all videos, other numbers: target videos
    # for MOT17 train/val, select in [2, 4, 5, 9, 10, 11, 13]
    # for MOT17 test, select in [1, 3, 6, 7, 8, 12, 14]
    # for MOT20 train/val, select in [1, 2, 3, 5]
    # for MOT20 test, select in [4, 6, 7, 8]
    parser.add_argument('--target_vid', type=int, default=target_vid, nargs='+')

    target_det = ['DPM', 'FRCNN', 'SDP']  # for MOT17, select in ['DPM', 'FRCNN', 'SDP']
    target_det = 'FRCNN'
    parser.add_argument('--target_det', type=str, default=target_det, nargs='+')

    # Arguments for detector
    det_cfg_file = 'reference_tracker'  # file name of target config in detector_cfgs directory
    parser.add_argument('--det_cfg_file', type=str, default=det_cfg_file)

    # Arguments for reference tracker
    # {0: 'sort', 1: 'deep_sort', 2: 'byte', 3: 'ocsort', 4: 'deep_ocsort'}
    parser.add_argument('--select_tracker', type=int, default=4)

    # General arguments for inference
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--vis_progress_bar', action='store_true', default=True)
    parser.add_argument('--run_name', type=str, default='deepocsort_origin')
    parser.add_argument('--vis_det', action='store_true', default=True)
    parser.add_argument('--vis_trk', action='store_true', default=True)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--view_size', type=int, default=[720, 1280], nargs='+')  # [height, width]
    parser.add_argument('--save_pred', action='store_true', default=True)
    parser.add_argument('--save_vid', action='store_true', default=False)
    parser.add_argument('--save_vid_fps', type=int, default=20)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FILE = Path(__file__).absolute()
    warnings.filterwarnings("ignore")
    np.set_printoptions(linewidth=np.inf)
    opt = get_args()
    main(opt)
