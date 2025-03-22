import cv2
from minio import Minio
from io import BytesIO
from kafka import KafkaProducer
from json import dumps
import logging
import os
import time
from elements.AlarmElement import AlarmElement

anomaly_logger = logging.getLogger(__name__)
NGINX_SOCKET_MINIO = os.environ.get("NGINX_SOCKET_MINIO")


class AlarmProducerNode:
    def __init__(self, config: dict) -> None:
        self.minio_config = config["minio"]
        self.kafka_config = config["kafka"]
        self.description_alarms = config["description_alarms"]

        self.minio_client = self._create_minio_client()
        self.last_send_timestamp_minio = None

        self.topic_name = self.kafka_config["topic_name"]
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=self.kafka_config["bootstrap_servers"],
            value_serializer=lambda x: dumps(x).encode("utf-8"),
        )

    def process(self, alarm_element: AlarmElement):
        image, anomalies, timestamp, source = (
            alarm_element.image,
            alarm_element.anomalies,
            alarm_element.timestamp,
            alarm_element.source,
        )
        timestamp = time.time()
        if self.last_send_timestamp_minio is not None:
            time_delta = timestamp - self.last_send_timestamp_minio
            if time_delta < self.minio_config["minio_freeze_time_secs"]:
                return

        image_path = self._generate_image_path_minio(anomalies, timestamp, source)
        description, priority_score = self._genetrate_description(anomalies)
        img_url = self._send_image_to_minio(image, image_path)
        self._send_kafka_message(img_url, description, priority_score)

        self.last_send_timestamp_minio = timestamp

    def _create_minio_client(self):
        minio_client = Minio(
            self.minio_config["socket"],
            access_key=self.minio_config["access_key"],
            secret_key=self.minio_config["secret_key"],
            secure=False,
        )
        if not minio_client.bucket_exists(self.minio_config["bucket_name"]):
            minio_client.make_bucket(self.minio_config["bucket_name"])

        return minio_client

    def _send_kafka_message(self, img_url, description, priority_score):
        message = {
            "img_url": str(img_url),
            "description": str(description),
            "priority_score": float(priority_score),
        }
        print("message", message)
        self.kafka_producer.send(
            self.topic_name,
            value=message,
        )

    def _genetrate_description(self, anomalies):
        priority_score = max(
            [self.description_alarms[anomaly]["priority_score"] for anomaly in anomalies]
        )
        anomaly_descriptions = [self.description_alarms[anomaly]["name"] for anomaly in anomalies]
        anomaly_descriptions = list(set(anomaly_descriptions))
        return ", ".join(anomaly_descriptions), priority_score

    def _generate_image_path_minio(self, anomalies, timestamp, source):

        img_basename = "_".join(anomalies)
        image_path = f"{source}/{int(timestamp)}_{img_basename}.jpeg"

        return image_path

    def _send_image_to_minio(self, image, img_path):
        buffer = cv2.imencode(".jpeg", image)[1]

        self.minio_client.put_object(
            self.minio_config["bucket_name"],
            object_name=img_path,
            data=BytesIO(buffer),
            length=len(buffer),
            content_type="image/jpeg",
        )

        url = self.minio_client.presigned_get_object(self.minio_config["bucket_name"], img_path)

        if "minio" in url and NGINX_SOCKET_MINIO is not None:
            url = url.replace("minio:9000", NGINX_SOCKET_MINIO)

        anomaly_logger.info(f"sent anomaly to {url}")
        return url
