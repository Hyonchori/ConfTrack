# Make detection prediction on MOT17/MOT20 dataset for using in Tracking by Detection Algorithm

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
from custom_utils.general_utils import increment_path, xywh2xyxy, xyxy2xywh
from custom_utils.plot_utils import letterbox, plot_info, plot_detection

from datasets.MOT.mot_dataset import get_mot_videos, MOT_CLASSES

from detector.config_detector import get_detector
from detector.detector_utils import scale_coords, clip_coords

from tracker.detection.config_detection import get_detections


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

    # Arguments for tracker configuration
    trk_cfg_file = args.trk_cfg_file
    sys.path.append((os.path.join(os.path.dirname(FILE), 'cfgs')))
    trk_cfg_file = importlib.import_module(trk_cfg_file)
    trk_cfg = trk_cfg_file.TrackerCFG()

    # General arguments for inference
    device = select_device(args.device)
    vis_progress_bar = args.vis_progress_bar
    run_name = args.run_name
    vis_det = args.vis_det
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
    out_dir = f'{FILE.parents[0]}/results/detector/{trk_cfg.type_detector}/{target_select}_{target_split}'
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

    # iterate videos
    start_time = time.time()
    for vid_idx, mot_dataset in enumerate(mot_videos):
        vid_name = mot_dataset.dataset.dataset_name
        print(f"\n--- Processing {vid_idx + 1} / {len(mot_videos)}'s video: {vid_name}")

        # create iterator on current video
        time.sleep(0.5)
        iterator = tqdm(enumerate(mot_dataset), total=len(mot_dataset),
                        desc=f'{vid_idx + 1}/{len(mot_videos)}: {vid_name}') \
            if vis_progress_bar else enumerate(mot_dataset)

        if save_vid:
            save_vid_path = os.path.join(save_dir, f'{vid_name}.mp4')
            vid_writer = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         save_vid_fps, (view_size[1], view_size[0]))

        mot_det_pred = ''
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
                if det is not None:
                    det = det.cpu().numpy()
                    scale_coords(detector.input_size, det, img_raw[0].shape[:2], center_pad=False)
                    if target_select == 'MOT20':
                        clip_coords(det, img_raw[0].shape[:2])
                else:
                    det = np.empty((0, 6))
            else:
                det = det[0]
                if len(det) != 0:
                    det = xywh2xyxy(det)
                else:
                    det = np.empty((0, 6))

            detections = get_detections(trk_cfg, det)
            te_det = time.time()

            # write detection results
            if save_pred:
                if len(det):
                    bboxes = xyxy2xywh(det)
                    for bbox in bboxes:
                        if bbox[-1] != 0:  # only save pedestrian
                            continue
                        if len(bbox) == 7:
                            mot_det_pred += f'{i + 1},-1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4] * bbox[5]}\n'
                        else:
                            mot_det_pred += f'{i + 1},-1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]}\n'

            # visualize detection
            ts_vis = time.time()
            if visualize:
                plot_info(img_v, f'{vid_name}: {i + 1} / {len(mot_dataset)}',
                          font_size=2, font_thickness=2)

                if vis_det:
                    plot_detection(img_v, detections, MOT_CLASSES, hide_cls=True, hide_confidence=False)

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
                vis_time = te_vis - ts_vis
                iter_time = te_vis - te_load
                print(f'load_time: {load_time:.4f}')
                print(f'det_time: {det_time:.4f}')
                print(f'vis_time: {vis_time:.4f}')
                print(f'\titer_total_time: {iter_time:.4f}')
            ts_load = time.time()

        if visualize:
            cv2.destroyWindow(vid_name)

        if save_pred:
            if target_select == 'MOT17':
                pred_save_path = os.path.join(save_dir,
                                              '-'.join(mot_dataset.dataset.dataset_name.split('-')[:-1]) + '.txt')
            else:  # target_select == 'MOT20'
                pred_save_path = os.path.join(save_dir,
                                              mot_dataset.dataset.dataset_name.split('.')[0] + '.txt')
            with open(pred_save_path, 'w') as f:
                f.write(mot_det_pred)

    if save_pred:
        print(f'\ntrack prediction results are saved in "{save_dir}"!')

    end_time = time.time()
    print(f'\nElapsed time: {end_time - start_time:.2f}')


def get_args():
    parser = argparse.ArgumentParser()

    # Arguments for MOT17/MOT20 dataset
    mot_root = '/home/jhc/Desktop/dataset/open_dataset/MOT'  # path to MOT dataset
    parser.add_argument('--mot_root', type=str, default=mot_root)

    target_select = 'MOT20'  # select in ['MOT17', 'MOT20']
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

    # Arguments for tracker
    trk_cfg_file = 'conftrack_det'  # file name of target config in tracker_cfgs directory
    parser.add_argument('--trk_cfg_file', type=str, default=trk_cfg_file)

    # General arguments for inference
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--vis_progress_bar', action='store_true', default=True)
    parser.add_argument('--run_name', type=str, default='yolox_x_coco_custom')
    parser.add_argument('--vis_det', action='store_true', default=True)
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
