import os
import glob
import json
import argparse

from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

from ultralytics import YOLO
from assets.dataloader_cliff import dataset_img
from models.cliff_hr48.cliff import CLIFF as cliff_hr48

import config
import constants

def bbox_extraction(vid_path, step=1):
    ## extracting bbox for the video subject (one subject only)

    detector = YOLO('./assets/checkpoint/yolov8x.pt')

    anno = {}
    anno['impath'] = []
    anno['bboxs_det']  = []
    anno['vid_start']  = []
    vid_start = 0
    anno['vid_start'].append(vid_start)

    frame_files = sorted(glob.glob(vid_path + '/*'))
    for f, f_name in enumerate(tqdm(frame_files, desc='YOLO Pred', total=len(frame_files))):
        if f % step !=0:
            continue

        frame = cv2.imread(f_name)
            
        results = detector.track(frame, verbose=False, persist=True, classes=[0])
        bbox = results[0].boxes.xyxy[0].tolist()

        # # Visualize the results on the frame
        # annotated_frame = frame.copy()
        # cv2.rectangle(annotated_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255,0,0), thickness=2)

        # # Display the annotated frame
        # cv2.imshow("YOLOv8 Tracking", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        # annotation data
        vid_start = vid_start + 1
        anno['impath'].append(f_name)
        anno['bboxs_det'].append(bbox)
    anno['vid_start'].append(vid_start)

    anno['datapath'] = ''
    vid_name = vid_path.split('/')[-1]
    vid_bbox_path = './demo/%s.json' % vid_name
    with open(vid_bbox_path, 'w') as f:
        json.dump(anno, f)

    print('saved bbox to %s' % vid_bbox_path)
    return vid_bbox_path

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not any(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def cliff_pred_extraction(vid_bbox_path, batch_size):

    ## generate per-frame pose and shape estimation using CLIFF given the bbox information

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cliff = eval("cliff_hr48")
    model = cliff('./models/cliff_hr48/smpl_mean_params.npz').to(device)
    state_dict = torch.load('./models/cliff_hr48/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
        
    data_loader = DataLoader(dataset=dataset_img(vid_bbox_path), 
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=2)
    N = len(dataset_img(vid_bbox_path))

    with open(vid_bbox_path) as f:
        vid_data = json.load(f)

    vid_data['pred_pose'] = []
    vid_data['pred_beta'] = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc='CLIFF Pred', total=len(data_loader))):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        
            # Feed images in the network to predict camera and SMPL parameters
            norm_img = batch["norm_img"].to(device).float()
            center = batch["center"].to(device).float()
            scale = batch["scale"].to(device).float()
            img_h = batch["img_h"].to(device).float()
            img_w = batch["img_w"].to(device).float()
            focal_length = batch["focal_length"].to(device).float()
            curr_batch_size = norm_img.shape[0]

            cx, cy, b = center[:, 0], center[:, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
            bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

            pred_pose, pred_betas, pred_cam_crop = model(norm_img, bbox_info)

            vid_data['pred_pose'].extend(pred_pose.detach().cpu().numpy().tolist())
            vid_data['pred_beta'].extend(pred_betas.detach().cpu().numpy().tolist())

        assert len(vid_data['pred_pose']) == len(vid_data['impath'])

        vid_processed_path = vid_bbox_path.split('.json')[0] + '_CLIFF.json'
        with open(vid_processed_path, 'w') as f:
            json.dump(vid_data, f)

    print('saved cliff pose and shape to %s' % vid_processed_path)
    return vid_processed_path

parser = argparse.ArgumentParser()
parser.add_argument('--vid_path', type=str, required=True, help='Path to the video folder')
parser.add_argument('--batch_zie', type=int, default=64, help='Batch size when performing inference')
parser.add_argument('--gpu', type=str, default='0', help='GPU to be used')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():

    vid_bbox_path = bbox_extraction(args.vid_path)

    vid_processed_path = cliff_pred_extraction(vid_bbox_path, batch_size=args.batch_zie)

if __name__ == '__main__':
    main()
