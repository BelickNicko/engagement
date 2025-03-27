import logging
import cv2
import time
import numpy as np
from collections import deque
from elements.FrameElement import FrameElement
from elements.AlarmElement import AlarmElement

logger = logging.getLogger(__name__)


class Statistic:
    def __init__(self, config: dict) -> None:
        statistic_config = config["Statistic"]

        self.how_often_seconds_check = statistic_config["how_often_seconds_check"]
        self.blinks_treshold_sleep_status = statistic_config["blinks_treshold_sleep_status"]
        self.period_to_set_sleep_status = statistic_config["period_to_set_sleep_status"]
        self.timestamp_eyes_opened = time.time()
        self.previus_time_blink_status_check = 0
        self.prev_sleep_status = 0
        self.eyes_were_closed = False
        self.previus_blinking_frequency = 0
        self.time_eyes_closed = 0
        self.video_fps = 30
        self.blinks_frequency_list = deque(maxlen=1500)
        self.current_time = None

    def process(self, frame_element: FrameElement) -> tuple[FrameElement, AlarmElement | None]:
        # Шаг 1: Обновление временной метки
        self.update_current_time(frame_element)

        # Шаг 2: Проверка необходимости обновления статусов
        if self.should_update_blink_status(frame_element):
            self.process_eye_status(frame_element)
            self.calculate_sleep_status(frame_element)
        else:
            self.set_previous_statuses(frame_element)

        # Шаг 3: Генерация предупреждений
        alarm_element = self.generate_alarms(frame_element)

        return frame_element, alarm_element

    def update_current_time(self, frame_element: FrameElement) -> None:
        """Обновляет текущее время на основе источника фрейма."""
        if isinstance(frame_element.source, int):
            self.current_time = frame_element.timestamp
        else:
            if self.current_time is None:
                self.current_time = 0
            else:
                self.current_time += 1 / self.video_fps

    def should_update_blink_status(self, frame_element: FrameElement) -> bool:
        """Проверяет, нужно ли обновлять статус моргания."""
        return (self.current_time - self.previus_time_blink_status_check >= 0.025) and (
            0 in frame_element.yolo_detected_human or len(frame_element.yolo_detected_human) == 0
        )

    def process_eye_status(self, frame_element: FrameElement) -> None:
        """Обрабатывает состояние открытых/закрытых глаз."""
        if frame_element.closed_eyes:
            if not self.eyes_were_closed:
                self.eyes_were_closed = True
                self.blinks_frequency_list.append((1, self.current_time))
        else:
            if self.eyes_were_closed:
                self.eyes_were_closed = False
                self.timestamp_eyes_opened = self.current_time
                self.blinks_frequency_list.append((0, self.timestamp_eyes_opened))

    def calculate_sleep_status(self, frame_element: FrameElement) -> None:
        """Рассчитывает статус сонливости."""
        if frame_element.frame_number > 30:
            statistic_blinks_window = [
                x[0] for x in self.blinks_frequency_list if self.current_time - x[1] < 60
            ]
            blinking_frequency = sum(statistic_blinks_window)

            try:
                percentage_share_of_blink = blinking_frequency / len(statistic_blinks_window)
            except ZeroDivisionError:
                print("Система разогревается")
                return

            # Определение состояния сонливости
            if percentage_share_of_blink > self.blinks_treshold_sleep_status:
                frame_element.sleep_status = 1
            else:
                frame_element.sleep_status = 0

            if self.current_time - self.timestamp_eyes_opened > self.period_to_set_sleep_status:
                frame_element.sleep_status = 1

            # Установка частоты морганий
            if not frame_element.sleep_status:
                frame_element.blinking_frequency = blinking_frequency
                self.previus_blinking_frequency = blinking_frequency
            else:
                frame_element.blinking_frequency = None
                self.previus_blinking_frequency = None

            self.prev_sleep_status = frame_element.sleep_status
            self.previus_time_blink_status_check = self.current_time

    def set_previous_statuses(self, frame_element: FrameElement) -> None:
        """Устанавливает предыдущие статусы, если не нужно обновлять."""
        if len(frame_element.yolo_detected_human) != 1:
            frame_element.blinking_frequency = None
            frame_element.sleep_status = None
        else:
            frame_element.blinking_frequency = self.previus_blinking_frequency
            frame_element.sleep_status = self.prev_sleep_status

    def generate_alarms(self, frame_element: FrameElement) -> AlarmElement | None:
        """Генерирует предупреждения (алармы)."""
        sleep_status_alarm = (
            "sleep_status_alarm"
            if frame_element.sleep_status == 1 and frame_element.frame_number > 300
            else None
        )

        human_out_of_frame_or_more_then_one_alarm = (
            "human_out_of_frame_or_more_then_one"
            if len(frame_element.yolo_detected_human) != 1 and frame_element.frame_number > 30
            else None
        )

        gadget_detection_alarm = (
            "gadget"
            if len(frame_element.yolo_detected_gadget) == 1 and frame_element.frame_number > 300
            else None
        )

        anomalies = [
            sleep_status_alarm,
            human_out_of_frame_or_more_then_one_alarm,
            gadget_detection_alarm,
        ]
        anomalies = list(filter(lambda x: x is not None, anomalies))

        if len(anomalies) > 0 and frame_element.frame_number > 600:
            return AlarmElement(
                frame_element.source,
                anomalies,
                frame_element.frame_result,
                frame_element.timestamp,
            )
        return None
