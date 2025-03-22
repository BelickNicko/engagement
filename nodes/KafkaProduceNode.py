from kafka import KafkaProducer
from json import dumps
import time
import numpy as np
from utils_local.utils import profile_time
from elements.VideoEndBreakElement import VideoEndBreakElement
from elements.FrameElement import FrameElement
import logging


class KafkaProducerNode:
    def __init__(self, config) -> None:
        config_kafka = config["kafka_producer_node"]
        bootstrap_servers = config_kafka["bootstrap_servers"]
        self.topic_name = config_kafka["topic_name"]
        self.how_often_sec = config_kafka["how_often_sec"]
        self.last_send_time = None
        time.sleep(3)
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: dumps(x).encode("utf-8"),
        )

    @profile_time
    def process(self, frame_element: FrameElement):
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element

        current_time = time.time()

        if frame_element.frame_number == 1:
            self.last_send_time = current_time

        if (
            current_time - self.last_send_time > self.how_often_sec
            or frame_element.frame_number == 1
        ):
            data = {
                "timestamp": frame_element.timestamp,
                "blinking_frequency": (
                    round(float(frame_element.blinking_frequency), 2)
                    if frame_element.blinking_frequency is not None
                    else None
                ),
                "sleep_status": (
                    frame_element.sleep_status if frame_element.sleep_status is not None else None
                ),
                "distance": (
                    frame_element.distance_to_the_object
                    if frame_element.distance_to_the_object is not None
                    else None
                ),
                "human_on_frame_status": (
                    len(list(map(lambda x: x == 1, frame_element.yolo_detected_human)))
                    if frame_element.yolo_detected_human is not None
                    else None
                ),
                "gadget_on_frame_status": (
                    int(len(frame_element.yolo_detected_gadget) > 0)
                    if frame_element.yolo_detected_gadget is not None
                    else None
                ),
            }
            self.kafka_producer.send(self.topic_name, value=data).get(timeout=1)
            logging.info(f"KAFKA sent message: {data} topic {self.topic_name}")
            self.last_send_time = current_time

        return frame_element
