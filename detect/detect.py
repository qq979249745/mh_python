import argparse
import time
from pathlib import Path

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Detect:
    def __init__(self):
        self.save_img = None
        self.colors = None
        self.names = None
        self.save_dir = None
        self.model = None
        self.half = None
        self.stride = None
        self.imgsz = None
        self.device = None
        self.opt = argparse.Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25,
                                      device='', exist_ok=False, img_size=640, iou_thres=0.45, name='',
                                      no_trace=False, nosave=False, project='../static/img', save_conf=False,
                                      save_txt=False, source='../static/img/1.jpg', update=False, view_img=False,
                                      weights='../best.pt')
        self.load_model(self.opt)

    def load_model(self, opt):
        with torch.no_grad():

            source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
            self.save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

            # Directories
            save_dir = Path(Path(opt.project), exist_ok=opt.exist_ok)  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            self.save_dir = save_dir
            # Initialize
            set_logging()
            device = select_device(opt.device)
            self.device = device
            half = device.type != 'cpu'  # half precision only supported on CUDA
            self.half = half
            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            self.model = model
            self.stride = int(model.stride.max())  # model stride
            self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

            if trace:
                model = TracedModel(model, device, opt.img_size)

            if half:
                model.half()  # to FP16

            # Get names and colors
            self.names = model.module.names if hasattr(model, 'module') else model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(device).type_as(next(model.parameters())))  # run once
            # Set Dataloader
            vid_path, vid_writer = None, None
            old_img_w = old_img_h = self.imgsz
            old_img_b = 1

    # def detect(self):
            t0 = time.time()
            dataset = LoadImages(self.opt.source, img_size=self.imgsz, stride=self.stride)
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
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
                t3 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(self.save_dir / p.name)  # img.jpg
                    txt_path = str(self.save_dir / 'labels' / p.stem) + ( '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
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
                            if self.opt.save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if self.save_img:  # Add bbox to image
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

            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.opt.save_txt else ''
            print(f"Results saved to {self.save_dir}{s}")

            print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='static/img/1.jpg', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='static/img', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = argparse.Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25,
                             device='', exist_ok=False, img_size=640, iou_thres=0.45, name='',
                             no_trace=False, nosave=False, project='static/img', save_conf=False,
                             save_txt=False, source='static/img/1.jpg', update=False, view_img=False, weights='best.pt')
    d = Detect()
    # d.detect()
