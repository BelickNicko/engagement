from kafka import KafkaProducer
from json import dumps
import time
import numpy as np
from utils_local.utils import profile_time
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

        current_time = time.time()

        if frame_element.frame_number == 1:
            self.last_send_time = current_time

        if (
            current_time - self.last_send_time > self.how_often_sec
            or frame_element.frame_number == 1
        ):
            movement_vectors = frame_element.movement_vectors
            if movement_vectors is not None:
                movement_vector_mean = np.mean(movement_vectors, axis=1)
                movement_vector_x = movement_vector_mean[0]
                movement_vector_y = movement_vector_mean[1]
            else:
                movement_vector_x = None
                movement_vector_y = None

            gaze_direction = frame_element.gaze_direction
            if gaze_direction is not None:
                gaze_direction_x = gaze_direction[0]
                gaze_direction_y = gaze_direction[1]
            else:
                gaze_direction_x = None
                gaze_direction_y = None
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
                "movement_vector_x": movement_vector_x,
                "movement_vector_y": movement_vector_y,
                "gaze_direction_x": gaze_direction_x,
                "gaze_direction_y": gaze_direction_y,
                "yaw": frame_element.yaw,
                "pitch": frame_element.pitch,
            }
            self.kafka_producer.send(self.topic_name, value=data).get(timeout=1)
            logging.info(f"KAFKA sent message: {data} topic {self.topic_name}")
            self.last_send_time = current_time

        return frame_element
