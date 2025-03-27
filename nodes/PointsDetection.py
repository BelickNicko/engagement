import time
import logging
import mediapipe as mp
import cv2
import numpy as np
from utils_local.utils import profile_time
from elements.FrameElement import FrameElement

logger = logging.getLogger(__name__)


class PointsDetection:
    def __init__(self, config: dict) -> None:
        config_PointsDetection = config["PointsDetection"]
        self.max_num_faces = config_PointsDetection["max_num_faces"]
        self.refine_landmarks = config_PointsDetection["refine_landmarks"]
        self.min_detection_confidence = config_PointsDetection["min_detection_confidence"]
        self.min_tracking_confidence = config_PointsDetection["min_tracking_confidence"]
        self.eye_idxs = config_PointsDetection["eye_idxs"]
        self.ear_tresh = config_PointsDetection["ear_tresh"]
        self.how_often_seconds = config_PointsDetection["how_often_seconds"]
        self.iris_coords = config_PointsDetection["iris_coords"]
        self.camera_focal_length = config_PointsDetection["camera_focal_length"]
        self.static_image_mode = True
        self.prev_movenets_check_time = time.time()
        # Инициализация mediapipe face mesh и drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        # Инициализация MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.previous_timestamp = 0

        self.optimal_distance_coef = 500  # мм

        self.iris_radius = 6
        self.iris_real_square = np.pi * self.iris_radius**2

        self.LEFT_IRIS = self.iris_coords["left"]
        self.RIGHT_IRIS = self.iris_coords["right"]

        self.previus_center_coords = None
        self.prev_movenet_vecrots = None

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:

        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionTrackingNodes | Неправильный формат входного элемента {type(frame_element)}"

        frame = frame_element.frame_result
        frame_h, frame_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            lanmarks = results.multi_face_landmarks[0].landmark
            left_eye_ear, left_eye_coords, left_eye_center = self.eye_coords(
                lanmarks, self.eye_idxs["left"], frame_w, frame_h
            )
            right_eye_ear, right_eye_coords, right_eye_center = self.eye_coords(
                lanmarks, self.eye_idxs["right"], frame_w, frame_h
            )

            iris_coords_array = self.iris_coords_det(lanmarks, frame_w, frame_h)
            frame_element.iris_coords_array = iris_coords_array
            if iris_coords_array is not None and (
                frame_element.timestamp - self.prev_movenets_check_time > 0.5
            ):
                current_center_coords = iris_coords_array[1::2]

                if self.previus_center_coords is not None:
                    movement_vectors = self.calculate_movement_vector(
                        current_center_coords, self.previus_center_coords
                    )
                    frame_element.movement_vectors = movement_vectors
                    self.prev_movenet_vecrots = movement_vectors
                self.previus_center_coords = iris_coords_array[1::2]
                self.prev_movenets_check_time = frame_element.timestamp
            else:
                frame_element.movement_vectors = self.prev_movenet_vecrots
            distance_to_the_object = self.distance_calculation(frame_element.iris_coords_array)
            frame_element.distance_to_the_object = int(distance_to_the_object)
            coef_dist_blink = self.optimal_distance_coef / distance_to_the_object

            frame_element.closed_eyes = int(
                any(
                    [
                        left_eye_ear < (self.ear_tresh * coef_dist_blink),
                        right_eye_ear < (self.ear_tresh * coef_dist_blink),
                    ]
                )
            )
            frame_element.detected_coords = left_eye_coords + right_eye_coords
            frame_element.eye_center_coords = [left_eye_center, right_eye_center]

            gaze_vector = self.calculate_gaze_direction(
                left_eye_center, right_eye_center, iris_coords_array
            )
            frame_element.gaze_direction = gaze_vector

            self.previous_timestamp = frame_element.timestamp

            frame_element.yaw, frame_element.pitch = self.calculate_head_pose(
                lanmarks, frame_w, frame_h
            )
        return frame_element

    def _denormalize_coordvinates(self, x, y, width, height):
        return int(x * width), int(y * height)

    def eye_coords(self, landmarks, refer_idxs, frame_width, frame_height):
        try:
            coords_points = []
            for i in refer_idxs:
                lm = landmarks[i]
                coord = self._denormalize_coordvinates(lm.x, lm.y, frame_width, frame_height)
                coords_points.append(coord)

            # Eye landmark (x, y)-coordinates
            P2_P6 = self._distance(coords_points[1], coords_points[5])
            P3_P5 = self._distance(coords_points[2], coords_points[4])
            P1_P4 = self._distance(coords_points[0], coords_points[3])

            # Вычисляем координату центра глаза (зрачка) как среднее значение координат
            eye_center = (
                int(sum(coord[0] for coord in coords_points) / len(coords_points)),
                int(sum(coord[1] for coord in coords_points) / len(coords_points)),
            )

            # Compute the eye aspect ratio
            ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

        except Exception as e:
            logger.error(f"Ошибка при вычислении координат глаза (зрачка): {e}")
            ear = 0
            coords_points = []
            eye_center = (0, 0)

        return ear, coords_points, eye_center

    def _distance(self, point_1, point_2):
        """Calculate l2-norm between two points"""
        dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
        return dist

    def iris_coords_det(self, lanmarks, frame_w, frame_h):

        left_eye_coords = []
        for lanmark_idx in self.LEFT_IRIS:
            lm = lanmarks[lanmark_idx]
            coord = self._denormalize_coordvinates(lm.x, lm.y, frame_w, frame_h)
            left_eye_coords.append(coord)

        right_eye_coords = []
        for lanmark_idx in self.RIGHT_IRIS:
            lm = lanmarks[lanmark_idx]
            coord = self._denormalize_coordvinates(lm.x, lm.y, frame_w, frame_h)
            right_eye_coords.append(coord)

        (l_cx, l_cy), iris_left_radius = cv2.minEnclosingCircle(np.array(left_eye_coords))
        (r_cx, r_cy), iris_right_radius = cv2.minEnclosingCircle(np.array(right_eye_coords))
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        return [iris_left_radius, center_left, iris_right_radius, center_right]

    def distance_calculation(self, iris_coords_array: np.ndarray):
        left_radius = iris_coords_array[0]
        right_radius = iris_coords_array[2]
        pixel_iris_square = (np.pi * (left_radius**2 + right_radius**2)) / 2
        real_distance = (
            (self.iris_real_square / pixel_iris_square) ** 0.5
            * 1000
            * (1 / self.camera_focal_length)
        )
        return real_distance

    def calculate_movement_vector(self, points_current, points_previous):
        # Проверяем, что количество точек совпадает
        if len(points_current) != len(points_previous):
            raise ValueError(
                "Количество точек на текущем и предыдущем кадрах должно быть одинаковым."
            )

        # Вычисляем разницу между координатами точек
        movement_vectors = []
        for current, previous in zip(points_current, points_previous):
            dx = current[0] - previous[0]
            dy = current[1] - previous[1]
            movement_vectors.append((dx, dy))

        return movement_vectors

    def calculate_gaze_direction(self, left_eye_center, right_eye_center, iris_coords_array):
        left_iris_center = iris_coords_array[1]
        right_iris_center = iris_coords_array[3]

        left_gaze_vector = (
            left_iris_center[0] - left_eye_center[0],
            left_iris_center[1] - left_eye_center[1],
        )
        right_gaze_vector = (
            right_iris_center[0] - right_eye_center[0],
            right_iris_center[1] - right_eye_center[1],
        )

        avg_gaze_vector = (
            (left_gaze_vector[0] + right_gaze_vector[0]) / 2,
            (left_gaze_vector[1] + right_gaze_vector[1]) / 2,
        )
        return avg_gaze_vector

    def calculate_head_pose(self, landmarks, frame_width, frame_height):
        # Получение координат ключевых точек

        nose_tip = landmarks[1]
        left_eye = landmarks[130]
        right_eye = landmarks[362]

        nose_tip = self._denormalize_coordvinates(nose_tip.x, nose_tip.y, frame_width, frame_height)
        left_eye = self._denormalize_coordvinates(left_eye.x, left_eye.y, frame_width, frame_height)
        right_eye = self._denormalize_coordvinates(
            right_eye.x, right_eye.y, frame_width, frame_height
        )

        vec_nose = np.array(nose_tip) - np.array(left_eye)

        # Угол рыскания (Yaw)
        yaw = np.arctan2(vec_nose[1], vec_nose[0]) * 180 / np.pi

        # Угол тангажа (Pitch)
        pitch = np.arctan2(vec_nose[0], vec_nose[1]) * 180 / np.pi

        return yaw, pitch
