from flask_server.VideoServer import VideoServer
from elements.FrameElement import FrameElement


class FlaskServerVideoNode:
    def __init__(self, config) -> None:
        self.video_server = VideoServer(config["host_ip"], config["port"])
        self.video_server.run()

    def process(self, frame_element: FrameElement):
        self.video_server.update_image(frame_element.frame_result)
