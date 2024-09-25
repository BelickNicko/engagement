from ultralytics import YOLO
import torch
from utils_local.utils import profile_time
from elements.VideoEndBreakElement import VideoEndBreakElement
from elements.FrameElement import FrameElement


class PersonDetectionNode:
    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция YOLO будет производиться на {device}')

        config_yolo = config["person_detection_node"]
        self.model = YOLO(config_yolo["weight_pth"], task="segment")
        self.classes = self.model.names
        self.conf = config_yolo["conf"]
        self.iou = config_yolo["iou"]
        self.imgsz = 640
        self.classes_to_detect = config_yolo["classes_to_detect"]
        self.use_only_cpu = config_yolo["use_only_cpu"]
        self.overlap_roi_threshold = config_yolo["overlap_roi_threshold"]
        self.time_duration = config_yolo["time_duration"]
        self.last_time_person_present = None

        if '.engine' in config_yolo["weight_pth"]:
            self.tensorrt = True
        else:
            self.tensorrt = False
            self.model.fuse()

    @profile_time
    def process(self, frame_element: FrameElement):
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionNode | Неправильный формат входного элемента {type(frame_element)}"

        frame = frame_element.frame_result.copy()
        if self.use_only_cpu:
            outputs = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
                                     iou=self.iou, classes=self.classes_to_detect, device='cpu')
        else:
            outputs = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
                                        iou=self.iou, classes=self.classes_to_detect)
        detected_conf = outputs[0].boxes.conf.cpu().tolist()

        if len(detected_conf) > 0:
            masks = outputs[0].masks.data
            masks = masks.cpu().numpy()  # Добавляем канал для изображения
            frame_element.person_masks = masks

        return frame_element
