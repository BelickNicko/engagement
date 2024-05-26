import os
import json
import time
import logging
from typing import Generator
import cv2

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement

logger = logging.getLogger(__name__)

class VideoReader: 

    def __init__(self, config: dict) -> None:
        config_VideoReader = config['VideoReader']
        self.video_pth = config_VideoReader["src"]
        self.video_source = f"Processing of {self.video_pth}"
        assert (
            os.path.isfile(self.video_pth) or type(self.video_pth) == int
        ), f"VideoReader| Файл {self.video_pth} не найден"

        self.stream = cv2.VideoCapture(self.video_pth)
        self.skip_secs = config_VideoReader["skip_secs"]
        self.start_timestamp = config_VideoReader["start_timestamp"]
        self.last_frame_timestamp = -1 # специально отрицательное значение при иницииализации
        self.first_timestamp = 0 # значение  времени в момент первого кадра потока

        self.break_element = False # проверяет отправлен ли элемент прерывающий видеопотока

        # устанавливаем ширину и высоту при обработке видео с камеры
        if type(self.video_pth) == int:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def process(self) -> Generator[FrameElement, None, None]:
        # номер кадра текущего видео
        frame_number = 0

        while True:
            ret, frame = self.stream.read()
            if not ret:
                logger.warning("Can't receive frame (stream end?). Exiting ...")
                if not self.break_element_sent:
                    self.break_element_sent = True
                    # отправляем VideoENdBreakElement чтобы обозначить окончание видеопотока
                    yield VideoEndBreakElement(self.video_pth, self.last_frame_timestamp)
                break
            # вычисляем timestemp в случае если если камера - стартуем с 0 секунд 
        # Вычисление timestamp в случае если вытягиваем с видоса или камеры (стартуем с 0 сек)
            if type(self.video_pth) == int:
                # с камеры:
                if frame_number == 0:
                    self.first_timestamp = time.time()
                timestamp = time.time() - self.first_timestamp
            else:
                # с видео:
                timestamp = self.stream.get(cv2.CAP_PROP_POS_MSEC) / 1000

                # делаем костыль, чтобы не было 0-вых тайстампов под конец стрима, баг cv2
                timestamp = (
                    timestamp
                    if timestamp > self.last_frame_timestamp
                    else self.last_frame_timestamp + 0.1
                )

            # Пропустим некоторые кадры если требуется согласно конфигу
            if abs(self.last_frame_timestamp - timestamp) < self.skip_secs:
                continue

            self.last_frame_timestamp = timestamp

            frame_number += 1
            if  type(self.video_pth) != int: #int - стрим с камеры, строка - запись видео
                time_date = timestamp + self.start_timestamp
            else:
                time_date = time.time() #timestamp
                
            yield FrameElement(self.video_source, frame, timestamp, frame_number, time_date)
