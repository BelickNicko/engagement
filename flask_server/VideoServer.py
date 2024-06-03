from flask import Flask, Response
from threading import Thread
import numpy as np
import cv2
import os


class VideoServer(object):
    app = None

    def __init__(self, host_ip, port):
        self.app = Flask(__name__)
        self.app.add_url_rule("/", "video", self._update_page)

        self.host_ip = host_ip
        self.port = port
        self._frame = np.zeros(shape=(640, 480), dtype=np.uint8)

        self.app_thread: Thread | None = None

    def _gen(self):
        while True:
            _, jpeg = cv2.imencode(".jpg", self._frame)
            encoded_image = jpeg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + encoded_image + b"\r\n\r\n")

    def _update_page(self) -> Response:
        return Response(self._gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def update_image(self, image: np.array):
        self._frame = image

    def run(self):
        self.app_thread = Thread(target=self.app.run, daemon=True, args=(self.host_ip, self.port))
        self.app_thread.start()


if __name__ == "__main__":
    # main для дебага
    video_server = VideoServer("localhost", 8008)
    video_server.run()
    while True:
        img = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        video_server.update_image(img)
