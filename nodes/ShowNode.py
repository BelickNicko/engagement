import logging
import cv2

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils_local.utils import profile_time, FPS_Counter
import math
import random
import numpy as np

logger = logging.getLogger(__name__)


class ShowNode:

    def __init__(self, config: dict) -> None:
        config_show_node = config["show_node"]
        self.scale = config_show_node["scale"]
        self.imshow = config_show_node["imshow"]
        self.fps_counter_N_frames_stat = config_show_node["fps_counter_N_frames_stat"]
        self.default_fps_counter = FPS_Counter(self.fps_counter_N_frames_stat)
        self.draw_fps_info = config_show_node["draw_fps_info"]
        self.show_coords = config_show_node["show_coords"]
        self.show_sleep_status = config_show_node["show_sleep_status"]
        self.show_irises = config_show_node["show_irises"]
        self.show_eye_center_vectos = config_show_node["show_eye_center_vectos"]
        # Параметры для шрифтов для статистики:
        self.fontFace = 1
        self.fontScale = 2.0
        self.thickness = 2

        # Параметры для шрифтов для alerta:
        self.fontFace_alert = 2
        self.fontScale_alert = 4
        self.thickness_alert = 2
        # Размеры и положение прямоугольника для текста
        self.rect_width = 400  # Ширина прямоугольника
        self.rect_height = 160  # Высота прямоугольника

        # отображение частоты морганий
        self.prev_blink_freq = (
            ""  # запоминаем предыдущее значение, чтобы выводить его между инференсами
        )

        self.prev_distance_to_the_object = ""

        # придется тут держать предыдущие точки
        self.previus_center_coords = None

    @profile_time
    def process(self, frame_element: FrameElement, fps_counter=None) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"ShowNode | Неправильный формат входного элемента {type(frame_element)}"

        frame_result = frame_element.frame_result.copy()
        image_height, image_width, _ = frame_result.shape
        # Подсчет fps и отрисовка
        if self.draw_fps_info:
            fps_counter = fps_counter if fps_counter is not None else self.default_fps_counter
            fps_real = fps_counter.calc_FPS()

            if frame_element.blinking_frequency == None:
                text = f"Blinks in minute: {self.prev_blink_freq}"
            else:
                text = "Blinks in minute: " + str(round(frame_element.blinking_frequency, 2))
                self.prev_blink_freq = str(round(frame_element.blinking_frequency, 2))

            if frame_element.distance_to_the_object == None:
                text += f" Distance meters: {self.prev_distance_to_the_object}"
            else:
                text += f" Distance meters: {frame_element.distance_to_the_object / 1000}"
                self.prev_distance_to_the_object = frame_element.distance_to_the_object / 1000

            text = f"FPS: {fps_real:.1f} {text}"

            (label_width, label_height), _ = cv2.getTextSize(
                text,
                fontFace=self.fontFace,
                fontScale=self.fontScale,
                thickness=self.thickness,
            )
            cv2.rectangle(
                frame_result, (0, 0), (10 + label_width, 35 + label_height), (0, 0, 0), -1
            )
            cv2.putText(
                img=frame_result,
                text=text,
                org=(10, 40),
                fontFace=self.fontFace,
                fontScale=self.fontScale,
                thickness=self.thickness,
                color=(255, 255, 255),
            )

        if self.show_coords and frame_element.detected_coords != []:
            self._draw_points(frame_result, frame_element.detected_coords)
            self._draw_points(frame_result, frame_element.eye_center_coords, color=(255, 0, 0))

        logger.info(f"/////////sleep_status: {frame_element.sleep_status}///////////")

        if self.show_sleep_status:
            if frame_element.sleep_status:

                alert = "Get up!"

                rect_x = (
                    image_width - self.rect_width
                ) // 2  # x-координата верхнего левого угла прямоугольника
                rect_y = (
                    image_height - self.rect_height
                ) // 2  # y-координата верхнего левого угла прямоугольника

                # Положение текста внутри прямоугольника
                text_x = rect_x + 10  # x-координата текста
                text_y = (
                    rect_y + self.rect_height // 2
                )  # y-координата текста (примерно посередине прямоугольника)

                # Отрисовка прямоугольника
                cv2.rectangle(
                    frame_result,
                    (rect_x, rect_y),
                    (rect_x + self.rect_width, rect_y + self.rect_height),
                    (0, 0, 255),
                    -1,
                )

                # Отрисовка текста
                cv2.putText(
                    img=frame_result,
                    text=alert,
                    org=(text_x, text_y),
                    fontFace=self.fontFace_alert,
                    fontScale=2.0,  # Увеличиваем размер текста
                    thickness=self.thickness_alert,
                    color=(255, 255, 255),
                )
        if self.show_irises and frame_element.iris_coords_array is not None:
            iris_coords_array = frame_element.iris_coords_array
            cv2.circle(
                frame_result,
                iris_coords_array[1],
                int(iris_coords_array[0]),
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame_result,
                iris_coords_array[3],
                int(iris_coords_array[2]),
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if self.show_eye_center_vectos and frame_element.iris_coords_array is not None:
            current_center_coords = frame_element.iris_coords_array[1::2]
            movement_vectors = frame_element.movement_vectors
            if self.previus_center_coords is not None and movement_vectors is not None:
                self.visualize_movement_vectors(
                    frame_result,
                    self.previus_center_coords,
                    current_center_coords,
                    movement_vectors,
                )
            # print("movement_vectors", movement_vectors)
            self.previus_center_coords = current_center_coords
        frame_result = cv2.resize(frame_result, (800, 600))
        frame_element.frame_result = frame_result
        frame_show = cv2.resize(frame_result.copy(), (-1, -1), fx=self.scale, fy=self.scale)

        if self.imshow:
            cv2.imshow(frame_element.source, frame_show)
            cv2.waitKey(1)

        return frame_element

    def _draw_points(self, image, points, radius=3, color=(0, 255, 0), thickness=1):
        for point in points:
            cv2.circle(image, point, radius, color, thickness)

    def visualize_movement_vectors(self, frame, points_current, points_previous, movement_vectors):
        for (x_current, y_current), (x_previous, y_previous), (dx, dy) in zip(
            points_current, points_previous, movement_vectors
        ):
            cv2.arrowedLine(frame, (x_previous, y_previous), (x_current, y_current), (0, 255, 0), 2)

            cv2.circle(frame, (x_current, y_current), 5, (0, 0, 255), -1)

            cv2.circle(frame, (x_previous, y_previous), 5, (255, 0, 0), -1)
