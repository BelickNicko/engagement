import hydra
from nodes.VideoReader import VideoReader
from nodes.ShowNode import ShowNode
from nodes.PointsDetection import PointsDetection
from nodes.Statistic import Statistic
from nodes.FlaskServerVideoNode import FlaskServerVideoNode
from nodes.SendInfoDBNode import SendInfoDBNode

@hydra.main(version_base=None, config_path="configs", config_name="app_config")
def main(config) -> None:
    video_reader = VideoReader(config)
    show_node = ShowNode(config)
    points_detection = PointsDetection(config)
    statistic_node = Statistic(config)
    render_video_flask = config["pipeline"]["render_video_flask"]
    send_info_db = config["pipeline"]["send_info_db"]
    if send_info_db: #если записываем в БД
        send_info_db_node = SendInfoDBNode(config)
    if render_video_flask: 
        flask_server_video_node = FlaskServerVideoNode(config["flask_server_video_node"])

    for frame_element in video_reader.process():
        frame_element = points_detection.process(frame_element)
        frame_element = statistic_node.process(frame_element)             
        frame_element = show_node.process(frame_element)
        if send_info_db:
            frame_element = send_info_db_node.process(frame_element)
        if render_video_flask:
             flask_server_video_node.process(frame_element)

if __name__ == "__main__":
    main()