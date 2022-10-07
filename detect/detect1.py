import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel


class Detect:
    def __init__(self):
        opt = argparse.Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25,
                                 device='', exist_ok=False, img_size=640, iou_thres=0.45, name='',
                                 no_trace=False, nosave=False, project='static/img', save_conf=True,
                                 save_txt=False, source='static/img/1.jpg', update=False, view_img=False,
                                 weights='best.pt')
        self.opt = opt

        self.source, weights, view_img, self.save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        self.save_img = not opt.nosave and not self.source.endswith('.txt')  # save inference images

        # Directories
        self.save_dir = Path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        self.device = device
        self.half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, device, opt.img_size)

        if self.half:
            self.model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))  # run once

    def detect(self):
        json_data = []
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1
        t0 = time.time()
        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'w') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img:  # Add bbox to image
                            json_data.append({
                                'label': self.names[int(cls)],
                                'conf': f'{conf:.2f}',
                                'xyxy': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                            })
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {self.save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        return json_data

    def detect_binary(self,file):
        json_data = []
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1
        t0 = time.time()
        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        #for path, img, im0s, vid_cap in dataset:
        path=self.opt.source
        im0s = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)  # BGR
        # Padded resize
        img = letterbox(im0s, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # img.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'w') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img:  # Add bbox to image
                        json_data.append({
                            'label': self.names[int(cls)],
                            'conf': f'{conf:.2f}',
                            'xyxy': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        })
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results (image with detections)
            if self.save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        if self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {self.save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        return json_data


if __name__ == '__main__':
    with torch.no_grad():
        d = Detect()
        d.detect()
