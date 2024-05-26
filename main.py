import hydra
from nodes.VideoReader import VideoReader
from nodes.ShowNode import ShowNode
from nodes.PointsDetection import PointsDetection
from nodes.Statistic import Statistic

@hydra.main(version_base=None, config_path="configs", config_name="app_config")
def main(config) -> None:
    video_reader = VideoReader(config)
    show_node = ShowNode(config)
    points_detection = PointsDetection(config)
    statistic_node = Statistic(config)
    for frame_element in video_reader.process():
        frame_element = points_detection.process(frame_element)
        frame_element = statistic_node.process(frame_element)             
        frame_element = show_node.process(frame_element)
  
if __name__ == "__main__":
    main()