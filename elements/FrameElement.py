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
        time_date: int,
        blink: int| None = None,
        detected_coords: list[list] | None = None,
        frame_result: np.ndarray | None = None,
        blinking_frequency: float | None = None,
        sleep_status: bool = False,
    ) -> None:
        
        self.source = source
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.time_date = time_date
        self.blink = blink
        self.detected_coords = detected_coords
        self.frame_result = frame_result
        self.blinking_frequency = blinking_frequency
        self.sleep_status = sleep_status
