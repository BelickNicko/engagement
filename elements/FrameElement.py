import time
import numpy as np
from typing import List, Optional


class FrameElement:
    # Класс, который характеризует каждый кадр
    def __init__(
        self,
        source: str,
        frame_result: np.ndarray,
        timestamp: float,
        frame_number: int,
        closed_eyes: Optional[int] = None,
        detected_coords: Optional[List[List]] = [[0, 0], [0, 0]],
        eye_center_coords: Optional[List[List]] = [[0, 0], [0, 0]],
        blinking_frequency: Optional[float] = 0.0,
        sleep_status: int = 0,
        iris_coords_array: np.ndarray = None,
        distance_to_the_object: int = None,
        yolo_detected_human: list = [0],
        yolo_detected_gadget: list = None,
        movement_vectors: list = None,
    ) -> None:

        self.source = source
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.closed_eyes = closed_eyes
        self.detected_coords = detected_coords
        self.eye_center_coords = eye_center_coords
        self.frame_result = frame_result
        self.blinking_frequency = blinking_frequency
        self.sleep_status = sleep_status
        self.iris_coords_array = iris_coords_array
        self.distance_to_the_object = distance_to_the_object
        self.yolo_detected_human = yolo_detected_human
        self.yolo_detected_gadget = yolo_detected_gadget
        self.movement_vectors = movement_vectors
