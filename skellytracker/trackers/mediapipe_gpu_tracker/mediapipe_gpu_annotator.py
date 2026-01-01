import cv2
import numpy as np


###
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import logging
###
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS, FACEMESH_LEFT_IRIS
from numpydantic import NDArray, Shape

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.mediapipe_gpu_tracker.mediapipe_gpu_observation import MediapipeGPUObservation

logger = logging.getLogger(__name__)

class MediapipeGPUAnnotatorConfig(BaseImageAnnotatorConfig):
    show_tracks: int | None = 15
    show_overlay: bool = True
    corner_marker_type: int = cv2.MARKER_DIAMOND
    corner_marker_size: int = 10
    corner_marker_thickness: int = 2
    corner_marker_color: tuple[int, int, int] = (0, 0, 255)

    aruco_lines_thickness: int = 2
    aruco_lines_color: tuple[int, int, int] = (0, 255, 0)

    text_color: tuple[int, int, int] = (215, 115, 40)
    text_size: float = .5
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX


class MediapipeGPUImageAnnotator(BaseImageAnnotator):
    config: MediapipeGPUAnnotatorConfig
    observations: list[MediapipeGPUObservation]

    @classmethod
    def create(cls, config: MediapipeGPUAnnotatorConfig):
        return cls(config=config, observations=[])

    def annotate_image(self, image, observation): 
        # logger.debug("Annotate entered")
        pose_landmarks = observation.pose_landmarks #MediapipeObservation.pose_landmarks
        if(len(pose_landmarks.landmark)>0):
            annotated_image = np.copy(image)
            # Draw the pose landmarks.
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
            return annotated_image
        else:
            logger.debug("no observation found this frame")
            return image # no observation, no markup.





