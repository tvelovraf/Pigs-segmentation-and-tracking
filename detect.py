import argparse
import os
import os.path as fs
import shutil
import time
from pathlib import Path

import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt

import math
import albumentations as albu
import segmentation_models_pytorch as smp
import sort.sort as Tracker

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box, plot_number_of_pigs)
from utils.torch_utils import select_device, time_synchronized

def validate_coords(x1, y1, x2, y2, w, h):
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= w:
        x2 = w-1
    if y2 >= h:
        y2 = h-1
    return x1,y1,x2,y2

def apply_SORT(tracker, detections_per_frame):
    if detections_per_frame is not None:
        if len(detections_per_frame) > 0:
            tracked_objects = tracker.update(detections_per_frame)
        else:
            tracked_objects = tracker.update(np.empty((0, 5)))

    return tracked_objects

def infer_segm(model, crop):
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    w_cr, h_cr = crop.shape[1], crop.shape[0]

    tr = albu.Compose([albu.augmentations.geometric.resize.Resize(320, 320)])
    crop = tr(image=crop)['image']

    tr = albu.Compose([
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor)])
    crop = tr(image=crop)['image']

    x_tensor = torch.from_numpy(crop).to('cuda').unsqueeze(0)

    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = cv2.resize(pr_mask, (w_cr, h_cr))

    return pr_mask


def apply_mask(frame, bbox, crop_mask, color=[0, 255, 255]):
    max_mask_value = np.max(crop_mask)
    idxs = crop_mask == max_mask_value
    frame[bbox[1]:bbox[3], bbox[0]:bbox[2], 0][idxs] = color[0]
    frame[bbox[1]:bbox[3], bbox[0]:bbox[2], 1][idxs] = color[1]
    frame[bbox[1]:bbox[3], bbox[0]:bbox[2], 2][idxs] = color[2]

    return frame

def detect(save_img=False):
    out, source = opt.output, opt.source

    imgsz = 1024
    conf_thres = 0.6
    weights_detect = './detect.pt'
    weights_segmentation = './segmentation.pth'
    iou_thres = 0.5

    # Initialize
    device = select_device('0')
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_detect = attempt_load(weights_detect, map_location=device)  # load FP32 model
    model_segmentation = torch.load(weights_segmentation)

    imgsz = check_img_size(imgsz, s=model_detect.stride.max())  # check img_size
    if half:
        model_detect.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model_detect.module.names if hasattr(model_detect, 'module') else model_detect.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model_detect(img.half() if half else img) if device.type != 'cpu' else None  # run once
    mot_tracker = Tracker.Sort()
    pigs_colors = {}
    pigs_activity = {}
    pigs_past_coordinates = {}
    pigs_activity_for_graphs = {}
    num_of_frame = -1
    for path, img, im0s, vid_cap in dataset:
        num_of_frame += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model_detect(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, agnostic=False)
        t2 = time_synchronized()

        # Process detections
        detections = []
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    __x1, __y1, __x2, __y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    detections.append([__x1, __y1, __x2, __y2, float(conf)])
                    # plot_one_box(xyxy, im0, label=label + '    '  + str(float(conf)), color=[0,0,255], line_thickness=2)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

        # Sort
        tracks = apply_SORT(mot_tracker, np.array(detections))

        # Post Proccessing tracks
        plot_number_of_pigs(im0, number_of_pigs=len(tracks))
        im0_tr = im0.copy()
        for track in tracks:
            _x1, _y1, _x2, _y2, number = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            _x1, _y1, _x2, _y2 = validate_coords(_x1, _y1, _x2, _y2, w=im0.shape[1], h=im0.shape[0])
            c = [_x1 + (_x2 - _x1) // 2, _y1 + (_y2 - _y1) // 2]


            if number not in pigs_colors:
                pigs_colors[number] = [np.random.randint(0, 255) for _ in range(3)]
                pigs_past_coordinates[number] = c
                pigs_activity[number] = 0
                pigs_activity_for_graphs[number] = [0 for i in range(0, num_of_frame)]

            pigs_activity[number] += int(math.sqrt(math.pow(c[0] - pigs_past_coordinates[number][0], 2) + math.pow(c[1] - pigs_past_coordinates[number][1], 2)))
            # maximum_activity = pigs_activity[max(pigs_activity, key=pigs_activity.get)]
            # for key, value in pigs_activity.items():
            #     try:
            #         pigs_activity[key] = int(value/maximum_activity*100)
            #         pigs_activity_for_graphs[key].append(int(value/maximum_activity*100))
            #     except ZeroDivisionError:
            #         continue
            pigs_past_coordinates[number] = c

            img_segm = im0[_y1:_y2, _x1:_x2, :]
            mask = infer_segm(model_segmentation, img_segm)
            im0_tr = apply_mask(im0_tr, [_x1, _y1, _x2, _y2], mask, color=pigs_colors[number])
            plot_one_box(track[0:4], im0, label='Num.: ' + str(number) + '        Act.: ' + str(pigs_activity[number]) + '%', color=pigs_colors[number], line_thickness=2)
        cv2.addWeighted(im0_tr, 0.3, im0, 1-0.3, 0, im0)
        if len(pigs_activity) == 0:
            maximum_activity = 0
        else:
            maximum_activity = pigs_activity[max(pigs_activity, key=pigs_activity.get)]
        for key, value in pigs_activity.items():
            try:
                pigs_activity[key] = int(value / maximum_activity * 100)
                pigs_activity_for_graphs[key].append(int(value / maximum_activity * 100))
            except ZeroDivisionError:
                continue


        # Save new Video
        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer

            fourcc = 'mp4v'  # output video codec
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        vid_writer.write(im0)

    # print(pigs_activity_for_graphs)
    for key, value in pigs_activity_for_graphs.items():
        y_coord = value
        x_coord = [xx for xx in range(len(y_coord))]
        lines = plt.plot(x_coord, y_coord)
        plt.setp(lines, color=[col/255.0 for col in pigs_colors[key]], linewidth=1.0)
        plt.xlabel('frame, n')
        plt.ylabel('activity, %')
    plt.savefig(out + '/activity.png')

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
