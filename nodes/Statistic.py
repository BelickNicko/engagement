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
        self.timestamp_eyes_opened = time.time()  # время когда глаза последний раз были открыты
        self.previus_time_blink_status_check = 0
        self.prev_sleep_status = 0
        self.eyes_were_closed = False
        self.previus_blinking_frequency = 0
        self.time_eyes_closed = 0
        self.video_fps = 30
        self.blinks_frequency_list = deque(maxlen=1500)
        self.current_time = None

    def process(self, frame_element: FrameElement) -> FrameElement:

        if type(frame_element.source) is int:
            self.current_time = frame_element.timestamp
        else:
            if self.current_time is None:
                self.current_time = 0
            else:
                self.current_time += 1 / self.video_fps

        # обновляем статусы раз в 1/20 секунды, иначе записываем предыдущие статусы
        if (self.current_time - self.previus_time_blink_status_check >= 0.025) and (
            0 in frame_element.yolo_detected_human or len(frame_element.yolo_detected_human) == 0
        ):
            # если есть фиксация закрытых глаз и фрейм назад они были открыты, то в динамический буфер добавляется 1
            # если фрейм назад глаза тоже были закрыты, то пропускаем
            # если нет фиксации того, что глаза закрыты и фрейм назад они были закрыты,  то в динамический буфер добавляется 0
            if frame_element.closed_eyes:
                if not self.eyes_were_closed:
                    self.eyes_were_closed = True
                    self.blinks_frequency_list.append((1, self.current_time))
            else:
                if self.eyes_were_closed:
                    self.eyes_were_closed = False
                    self.timestamp_eyes_opened = self.current_time
                    self.blinks_frequency_list.append((0, self.timestamp_eyes_opened))
            # вторая часть ноды
            # заходим в нее только, когда система разогрелась: номер текущего фрейма видеопотока больше 30
            if frame_element.frame_number > 30:
                # отфильтровываем динамический буффер по времени, берем только последние 60 секунд
                statistic_blinks_window = [
                    x[0] for x in self.blinks_frequency_list if self.current_time - x[1] < 60
                ]
                # рассчитываем количество морганий за минуту и их процентаж
                blinking_frequency = sum(statistic_blinks_window)
                try:
                    percentage_share_of_blink = blinking_frequency / len(statistic_blinks_window)
                except ZeroDivisionError:
                    print("система разогревается")
                    return frame_element, None
                # проверяем, фиксируется ли состояние сонливости
                # если процентаж больше установленного порога, то фиксируем состояние сонливости
                if percentage_share_of_blink > self.blinks_treshold_sleep_status:
                    frame_element.sleep_status = 1
                else:
                    frame_element.sleep_status = 0

                # если время когда глаза не открыты больше порога, фиксируем состояние сонливости
                if self.current_time - self.timestamp_eyes_opened > self.period_to_set_sleep_status:
                    frame_element.sleep_status = 1
                self.prev_sleep_status = frame_element.sleep_status
                # в зависимости от статуса сонливости задаем частоту морганий
                if not frame_element.sleep_status:
                    frame_element.blinking_frequency = blinking_frequency
                    self.previus_blinking_frequency = blinking_frequency
                else:
                    frame_element.blinking_frequency = None
                    self.previus_blinking_frequency = None

            self.previus_time_blink_status_check = self.current_time

        else:
            if len(frame_element.yolo_detected_human) == 0:
                frame_element.blinking_frequency = None
                frame_element.sleep_status = None
            else:
                frame_element.blinking_frequency = self.previus_blinking_frequency
                frame_element.sleep_status = self.prev_sleep_status

        sleep_status_alarm = frame_element.sleep_status
        if sleep_status_alarm == 1 and frame_element.frame_number > 300:
            sleep_status_alarm = "sleep_status_alarm"
        else:
            sleep_status_alarm = None

        human_out_of_frame_or_more_then_one_alarm = frame_element.yolo_detected_human
        if len(human_out_of_frame_or_more_then_one_alarm) != 1 and frame_element.frame_number > 30:
            human_out_of_frame_or_more_then_one_alarm = "human_out_of_frame_or_more_then_one"
        else:
            human_out_of_frame_or_more_then_one_alarm = None

        gadget_detection_alarm = frame_element.yolo_detected_gadget
        if len(gadget_detection_alarm) == 1 and frame_element.frame_number > 300:
            gadget_detection_alarm = "gadget"
        else:
            gadget_detection_alarm = None

        anomalies = [
            sleep_status_alarm,
            human_out_of_frame_or_more_then_one_alarm,
            gadget_detection_alarm,
        ]
        anomalies = list(filter(lambda x: x is not None, anomalies))

        if len(anomalies) > 0 and frame_element.frame_number > 600:
            alarm_element = AlarmElement(
                frame_element.source,
                anomalies,
                frame_element.frame_result,
                frame_element.timestamp,
            )
        else:
            alarm_element = None
        return frame_element, alarm_element
