import os
import json
import time
import logging
from typing import Generator
import cv2

from elements.FrameElement import FrameElement

logger = logging.getLogger(__name__)


class VideoReader:

    def __init__(self, config: dict) -> None:
        config_VideoReader = config["VideoReader"]
        self.video_pth = config_VideoReader["src"]
        self.video_source = f"Processing of {self.video_pth}"
        assert (
            os.path.isfile(self.video_pth) or type(self.video_pth) == int
        ), f"VideoReader| Файл {self.video_pth} не найден"

        self.stream = cv2.VideoCapture(self.video_pth)
        self.skip_secs = config_VideoReader["skip_secs"]
        self.last_frame_timestamp = -1  # специально отрицательное значение при иницииализации
        self.first_timestamp = 0  # значение  времени в момент первого кадра потока

        self.break_element = False  # проверяет отправлен ли элемент прерывающий видеопотока

        # устанавливаем ширину и высоту при обработке видео с камеры
        if type(self.video_pth) == int:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def process(self) -> Generator[FrameElement, None, None]:
        # номер кадра текущего видео
        frame_number = 0

        while True:
            ret, frame = self.stream.read()
            if not ret:  # FIXME
                break

            timestamp = time.time()  # timestamp
            frame_number += 1

            # Пропустим некоторые кадры если требуется согласно конфигу
            if abs(self.last_frame_timestamp - timestamp) < self.skip_secs:
                continue
            if self.video_source == 0:
                self.video_source = "0"
            yield FrameElement(self.video_source, frame, timestamp, frame_number)
