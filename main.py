import hydra
from nodes.VideoReader import VideoReader
from nodes.ShowNode import ShowNode
from nodes.PointsDetection import PointsDetection
from nodes.Statistic import Statistic
from nodes.FlaskServerVideoNode import FlaskServerVideoNode
from nodes.KafkaProduceNode import KafkaProducerNode
from nodes.HumanGadgetDetection import PersonGadgetDetectionNode
from nodes.AlarmNode import AlarmProducerNode


@hydra.main(version_base=None, config_path="configs", config_name="app_config")
def main(config) -> None:
    video_reader = VideoReader(config)
    show_node = ShowNode(config)
    points_detection = PointsDetection(config)
    statistic_node = Statistic(config)
    if config["pipeline"]["detect_human"]:
        person_detection_node = PersonGadgetDetectionNode(config)
    render_video_flask = config["pipeline"]["render_video_flask"]
    send_to_kafka = config["pipeline"]["send_to_kafka"]

    if send_to_kafka:  # если записываем в БД
        send_to_kafka_node = KafkaProducerNode(config)
    if render_video_flask:
        flask_server_video_node = FlaskServerVideoNode(config["flask_server_video_node"])
    alarm_producer_node = AlarmProducerNode(config["alarm_producer_node"])

    for frame_element in video_reader.process():
        if config["pipeline"]["detect_human"]:
            frame_element = person_detection_node.process(frame_element)
        frame_element = points_detection.process(frame_element)
        frame_element, alarm_element = statistic_node.process(frame_element)
        if alarm_element is not None:
            alarm_producer_node.process(alarm_element)
        frame_element = show_node.process(frame_element)
        if send_to_kafka:
            frame_element = send_to_kafka_node.process(frame_element)
        if render_video_flask:
            flask_server_video_node.process(frame_element)


if __name__ == "__main__":
    main()
