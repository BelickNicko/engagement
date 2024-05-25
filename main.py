import hydra
from nodes.VideoReader import VideoReader
from nodes.ShowNode import ShowNode
@hydra.main(version_base=None, config_path="configs", config_name="app_config")

def main(config) -> None:
    video_reader = VideoReader(config["video_reader"])
    show_node = ShowNode(config)

    for frame_element in video_reader.process():
        frame_element = show_node.process(frame_element)

if __name__ == "__main__":
    main()