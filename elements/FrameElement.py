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
        frame_result: np.ndarray | None = None,
        
    ) -> None:
        
        self.source = source
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.frame_result = frame_result
