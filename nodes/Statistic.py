import logging
import cv2
import numpy as np
from collections import deque
from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement

logger = logging.getLogger(__name__)


class Statistic:
    def __init__(self, config: dict) -> None:
        statistic_config = config["Statistic"]
        self.buffer_size = statistic_config["buffer_size"]
        self.how_often_seconds = statistic_config["how_often_seconds"]
        self.blinking_buffer = []  # длина буфера для инференса
        self.previous_timestamp = 0
        self.prev_sleep_status = False

    def process(self, frame_element: FrameElement) -> FrameElement:

        self.blinking_buffer.append(frame_element.blink)  # накапливаем статус сонливости

        if (
            #len(self.blinking_buffer) >= self.buffer_size and 
            round(frame_element.timestamp) % self.how_often_seconds == 0
            and self.previous_timestamp != round(frame_element.timestamp)
        ):
            frame_element.blinking_frequency = np.sum(self.blinking_buffer) / len(self.blinking_buffer) 
            frame_element.sleep_status = np.percentile(self.blinking_buffer, 60) != 0
            print("blinking_buffer: ", len(self.blinking_buffer))
            print("delta: ", round(frame_element.timestamp - self.previous_timestamp))
            self.previous_timestamp = round(frame_element.timestamp)
            self.prev_sleep_status = frame_element.sleep_status
            self.blinking_buffer = [] 
        else:
            frame_element.sleep_status = self.prev_sleep_status

        return frame_element
