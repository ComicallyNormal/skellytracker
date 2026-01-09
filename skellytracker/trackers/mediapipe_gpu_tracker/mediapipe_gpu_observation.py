from copy import copy
from typing import NamedTuple

import logging
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import Landmark
from typing import List


from mediapipe.python.solutions import holistic as mp_holistic

from mediapipe.tasks.python.components.containers.landmark import (
    NormalizedLandmark,
    Landmark,
)


from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, \
    LandmarkList  # linter sees an error here, but it runs fine
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, \
    LandmarkList  # linter sees an error here, but it runs fine
from mediapipe.python.solutions.face_mesh import FACEMESH_NUM_LANDMARKS_WITH_IRISES

from numpydantic import NDArray, Shape

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation, TrackerTypeString, TrackedPoint2dArray
from skellytracker.trackers.mediapipe_tracker.get_mediapipe_face_info import MEDIAPIPE_FACE_CONTOURS_INDICIES, \
    MEDIAPIPE_FACE_CONTOURS_NAMES

logger = logging.getLogger(__name__)
MediapipeResults = NamedTuple

# TODO: use numpydantic to fix numpy type hints for this
class MediapipeGPUObservation(BaseObservation):
    tracker_type:TrackerTypeString = 'mediapipe_gpu_tracker'
    frame_number: int  # the frame number of the image in which this observation was made
    pose_landmarks: NormalizedLandmarkList
    image_size: tuple[int, int]

    
    def to_pb_normalized_landmark_list(pose_landmarks_list : list[NormalizedLandmark])->NormalizedLandmarkList:
        """
        pose_landmarks_list: list of Tasks NormalizedLandmark objects
        returns: landmark_pb2.NormalizedLandmarkList
        """
        out = NormalizedLandmarkList()
        for lm in pose_landmarks_list:
            out.landmark.add(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=getattr(lm, "visibility", 0.0),
                presence=getattr(lm, "presence", 0.0),
            )
        return out
    
    @classmethod
    def from_detection_results(cls,
                               frame_number: int,
                               mediapipe_results: MediapipeResults,
                               image_size: tuple[int, int],
                               include_segmentation_mask: bool = True):
        converted_landmarks = NormalizedLandmarkList()
        if(len(mediapipe_results.pose_landmarks)>0):
            converted_landmarks = cls.to_pb_normalized_landmark_list(mediapipe_results.pose_landmarks[0])
        else:
            pass
        if include_segmentation_mask:  # TODO: make sure we don't get a missing attribute error
            segmentation_mask = mediapipe_results.segmentation_masks
        else:
            segmentation_mask = None
        return cls(
            frame_number=frame_number,
            pose_landmarks=converted_landmarks,
            image_size=image_size
        )


    @property
    def body_landmark_names(self) -> list[str]:
        return [f"body.{landmark.name.lower()}" for landmark in mp_holistic.PoseLandmark]


    @property
    def num_body_points(self) -> int:
        return len(self.body_landmark_names)


    @property
    def num_total_points(self) -> int:
        return self.num_body_points

    @property
    def body_points_xyz(self) -> NDArray[Shape["* body points, 3"], float]:
        # logger.info("num of points: " + str(len(self.pose_landmarks.landmark)))
        if self.pose_landmarks is None or len(self.pose_landmarks.landmark)==0:
            return np.full((self.num_body_points, 3), np.nan)

        return self._landmarks_to_array(self.pose_landmarks)



    def _landmarks_to_array(self, landmarks: NormalizedLandmarkList) -> NDArray[Shape["* all points, 3"], float]:
        landmark_array = np.array(
            [
                (landmark.x, landmark.y, landmark.z)
                for landmark in landmarks.landmark
            ]
        )

        # convert from normalized image coordinates to pixel coordinates
        landmark_array *= np.array([self.image_size[1], self.image_size[0],self.image_size[1]])  # multiply z by image width per mediapipe docs

        return landmark_array

    def all_points(self, dimensions:int, face_type: str = "contour",  scale_by:float=1.0) -> dict[str, tuple]:
        if not dimensions in [2, 3]:
            raise ValueError(f"Invalid dimensions: {dimensions}")

        all_points_by_name = {}
        body_xyz = self.body_points_xyz.copy()* scale_by

        for index, point_name in enumerate(self.body_landmark_names):
            all_points_by_name[point_name] = tuple(body_xyz[index, :dimensions])

        return all_points_by_name

    def get_confidence_scores(self) -> NDArray[Shape["* number_of_points"], float] | None:
        """
        Get visibility scores for all tracked points (MediaPipe's confidence metric).

        Returns:
            Array of visibility scores for body, hands, and face points
        """
        body_visibility = self._get_body_visibility()
        # right_hand_visibility = self._get_hand_visibility(self.right_hand_landmarks)
        # left_hand_visibility = self._get_hand_visibility(self.left_hand_landmarks)
        # face_visibility = self._get_face_visibility()

        return np.concatenate([
            body_visibility
        ])

    def _get_body_visibility(self) -> NDArray[Shape["* body points"], float]:
        """Extract visibility scores from body landmarks."""
        if self.pose_landmarks is None:
            return np.full(self.num_body_points, 0.0)

        return np.array([
            landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            for landmark in self.pose_landmarks.landmark
        ])

    def _get_hand_visibility(self, hand_landmarks: NormalizedLandmarkList | None) -> NDArray[
        Shape["* hand points"], float]:
        """Extract visibility scores from hand landmarks."""
        if hand_landmarks is None:
            return np.full(self.num_single_hand_points, 0.0)

        # MediaPipe hand landmarks typically don't have visibility, use presence (0 or 1)
        return np.array([
            landmark.presence if hasattr(landmark, 'presence') else 1.0
            for landmark in hand_landmarks.landmark
        ])

    def _get_face_visibility(self) -> NDArray[Shape["* face points"], float]:
        """Extract visibility scores from face landmarks."""
        if self.face_landmarks is None:
            return np.full(self.num_face_contour_points, 0.0)

        face_contour_indices = list(MEDIAPIPE_FACE_CONTOURS_INDICIES)

        # Check if we have any face data
        if np.isnan(self.face_tesselation_points_xyz).all():
            return np.full(len(face_contour_indices), 0.0)

        # Get visibility for contour points only
        visibilities = []
        for idx in face_contour_indices:
            if idx < len(self.face_landmarks.landmark):
                landmark = self.face_landmarks.landmark[idx]
                visibility = landmark.presence if hasattr(landmark, 'presence') else 1.0
            else:
                # Iris landmarks that might be missing
                visibility = 0.0
            visibilities.append(visibility)

        return np.array(visibilities)

    def to_2d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> NDArray[
        Shape["33, 2"], float]:
        """
        Convert to 2D array with optional confidence filtering.

        Args:
            confidence_threshold: Minimum visibility to include point. If None, no filtering.
            fill_with_nans: Whether to fill low-confidence points with NaN.
        """
        # points_2d = np.concatenate(
        #     (
        #         self.body_points_xyz[..., :2]
        #     ),
        #     axis=0,
        # )

        points_2d = np.asarray(self.body_points_xyz)[..., :2]  # (33,2)


        if confidence_threshold is not None:
            confidence_scores = self.get_confidence_scores()
            points_2d = self.filter_by_confidence(
                points=points_2d,
                confidence_scores=confidence_scores,
                confidence_threshold=confidence_threshold,
                fill_with_nans=fill_with_nans
            )

        return points_2d

    def to_3d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> NDArray[
        Shape["33, 3"], float]:
        """
        Convert to 3D array with optional confidence filtering.

        Args:
            confidence_threshold: Minimum visibility to include point. If None, no filtering.
            fill_with_nans: Whether to fill low-confidence points with NaN.
        """
        points_3d = np.concatenate(
            (
                self.body_points_xyz
            ),
            axis=0,
        )

        if confidence_threshold is not None:
            confidence_scores = self.get_confidence_scores()
            points_3d = self.filter_by_confidence(
                points=points_3d,
                confidence_scores=confidence_scores,
                confidence_threshold=confidence_threshold,
                fill_with_nans=fill_with_nans
            )

        return points_3d

    def to_tracked_points(self, *, confidence_threshold: float | None = None) -> dict[str, TrackedPoint2dArray]:
        """Get tracked points filtered by confidence."""
        points = self.all_points(dimensions=2)

        if confidence_threshold is not None:
            confidence_scores = self.get_confidence_scores()

            # Build mapping of point names to confidence scores
            all_names = (
                    self.body_landmark_names
            )

            filtered_points = {}
            for i, name in enumerate(all_names):
                if confidence_scores[i] >= confidence_threshold:
                    if name in points:
                        filtered_points[name] = np.array(points[name])

            return filtered_points

        return {name: np.array([x, y]) for name, (x, y) in points.items()}





MediapipeObservations = list[MediapipeGPUObservation]
