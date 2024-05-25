import os
import json
import time
import logging
from typing import Generator
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

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
        self.static_image_mode = True
        # Инициализация mediapipe face mesh и drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_facemesh = mp.solutions.face_mesh
        
    def process(self, frame_element: FrameElement) -> None:        
        
        #mp_drawing = mp.solutions.drawing_utils
        self.mp_facemesh = mp.solutions.face_mesh
        # Инициализация MediaPipe Face Mesh
        with self.mp_facemesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.min_detection_confidence) as face_mesh:
        # Обработка изображения с Face Mesh
            results = face_mesh.process(frame_element)

            # Проверка, нашлись ли лица
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Объединение всех индексов в один список
                specific_idxs = eye_idxs["left"] + eye_idxs["right"]
                
                # Вызов функции для отрисовки конкретных точек
                plot_specific_landmarks(
                    img_dt=image,
                    face_landmarks=face_landmarks,
                    specific_idxs=specific_idxs
                )
            else:
                print("Лицо не найдено на изображении.")
    def _denormalize_coordinates(x, y, width, height):
        return int(x * width), int(y * height)