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
        time_date: int,
        blink: Optional[int] = None,
        detected_coords: Optional[List[List]] = None,
        blinking_frequency: Optional[float] = None,
        sleep_status: bool = False,
    ) -> None:
        
        self.source = source
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.time_date = time_date
        self.blink = blink
        self.detected_coords = detected_coords
        self.frame_result = frame_result
        self.blinking_frequency = blinking_frequency
        self.sleep_status = sleep_status
