import time 
import numpy as np

class FrameElement:
    # Класс, который характеризует каждый кадр
    def __init__(
        self,
        source: str,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int,
        sleep_status: int| None = None,
        detected_coords: list[list] | None = None,
        frame_result: np.ndarray | None = None,
        
    ) -> None:
        
        self.source = source
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.sleep_status = sleep_status
        self.detected_coords = detected_coords
        self.frame_result = frame_result
