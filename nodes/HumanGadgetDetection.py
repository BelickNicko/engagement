from ultralytics import YOLO
import torch
from utils_local.utils import profile_time
from elements.FrameElement import FrameElement
import logging
import time

logger = logging.getLogger(__name__)


class PersonGadgetDetectionNode:
    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Детекция YOLO будет производиться на {device}")

        config_yolo = config["person_detection_node"]
        self.model = YOLO(config_yolo["weight_pth"], task="detect")
        self.classes = self.model.names
        self.conf = config_yolo["conf"]
        self.iou = config_yolo["iou"]
        self.imgsz = 224
        self.classes_to_detect = config_yolo["classes_to_detect"]
        self.use_only_cpu = config_yolo["use_only_cpu"]
        self.time_duration = config_yolo["time_duration"]
        self.prev_check_time = time.time()
        if ".engine" in config_yolo["weight_pth"]:
            self.tensorrt = True
        else:
            self.tensorrt = False
            self.model.fuse()
        self.prev_detected_human = []
        self.prev_detected_gadget = []

    @profile_time
    def process(self, frame_element: FrameElement):

        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionNode | Неправильный формат входного элемента {type(frame_element)}"

        if (
            time.time() - self.prev_check_time >= self.time_duration
            or frame_element.frame_number == 1
        ):
            frame = frame_element.frame_result.copy()
            if self.use_only_cpu:
                outputs = self.model.predict(
                    frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    verbose=False,
                    iou=self.iou,
                    classes=self.classes_to_detect,
                    device="cpu",
                )
            else:
                outputs = self.model.predict(
                    frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    verbose=False,
                    iou=self.iou,
                    classes=self.classes_to_detect,
                )
            resulted_class = outputs[0].boxes.cls.tolist()

            resulted_class_human = list(filter(lambda x: x == 0, resulted_class))
            yolo_detected_gadget = list(filter(lambda x: x == 67, resulted_class))

            frame_element.yolo_detected_human = resulted_class_human
            frame_element.yolo_detected_gadget = yolo_detected_gadget

            self.prev_detected_human = resulted_class_human
            self.prev_detected_gadget = yolo_detected_gadget

            self.prev_check_time = frame_element.timestamp
        else:
            frame_element.yolo_detected_human = self.prev_detected_human
            frame_element.yolo_detected_gadget = self.prev_detected_gadget

        return frame_element
