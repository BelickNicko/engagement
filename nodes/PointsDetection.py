import time
import logging
import mediapipe as mp
from collections import deque

from utils_local.utils import profile_time
from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement

logger = logging.getLogger(__name__)

class PointsDetection: 
    def __init__(self, config: dict) -> None:
        config_PointsDetection = config['PointsDetection']
        self.max_num_faces  = config_PointsDetection["max_num_faces"]
        self.refine_landmarks = config_PointsDetection["refine_landmarks"]
        self.min_detection_confidence = config_PointsDetection["min_detection_confidence"]
        self.min_tracking_confidence = config_PointsDetection["min_tracking_confidence"]
        self.eye_idxs = config_PointsDetection["eye_idxs"]
        self.ear_tresh  = config_PointsDetection["ear_tresh"]
        self.static_image_mode = True
        # Инициализация mediapipe face mesh и drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_facemesh = mp.solutions.face_mesh
          # Инициализация MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        
        
    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:        
        
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionTrackingNodes | Неправильный формат входного элемента {type(frame_element)}"

        frame = frame_element.frame_result
        frame_h, frame_w, _ = frame.shape

        results = self.face_mesh.process(frame)
        if results.multi_face_landmarks:
            lanmarks = results.multi_face_landmarks[0].landmark
            left_eye_ear, left_eye_coords =  self._coords(lanmarks, self.eye_idxs['left'], frame_w, frame_h)
            right_eye_ear, right_eye_coords =  self._coords(lanmarks, self.eye_idxs['right'], frame_w, frame_h)
            frame_element.blink = int(any([left_eye_ear < self.ear_tresh, right_eye_ear < self.ear_tresh]))

            
            frame_element.detected_coords = left_eye_coords + right_eye_coords
        else:
            logger.warning("Can't find out key points")
            frame_element.blink = -1
            frame_element.detected_coords  = []
                
        return frame_element
    
    def _denormalize_coordinates(self, x, y, width, height):
        return int(x * width), int(y * height)

    def _coords(self, lanmarks, refer_idxs, frame_width, frame_height):    
        try:
            coords_points = []
            for i in refer_idxs:
                lm = lanmarks[i]
                coord = self._denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
                coords_points.append(coord)

            # Eye landmark (x, y)-coordinates
            P2_P6 = self._distance(coords_points[1], coords_points[5])
            P3_P5 = self._distance(coords_points[2], coords_points[4])
            P1_P4 = self._distance(coords_points[0], coords_points[3])

            # Compute the eye aspect ratio
            ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

        except:
            ear = 0
            coords_points = []

        return ear, coords_points
    
    def _distance(self, point_1, point_2):
        """Calculate l2-norm between two points"""
        dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
        return dist