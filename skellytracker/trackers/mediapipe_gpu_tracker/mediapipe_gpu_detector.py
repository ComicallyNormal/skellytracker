from enum import Enum

import mediapipe as mp
import numpy as np
import cv2
import time
import logging

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.mediapipe_gpu_tracker.mediapipe_gpu_observation import MediapipeGPUObservation, MediapipeResults
#####NEW CODE
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
######

logger = logging.getLogger(__name__)


class MediapipeModelComplexity(int, Enum):
    LITE = 0  # BlazePose Lite model, fastest
    FULL = 1  # BlazePose Full model, balanced
    HEAVY = 2  # BlazePose Heavy model, most accurate


class MediapipeGPUDetectorConfig(BaseDetectorConfig):
    processor_type : python.BaseOptions.Delegate = python.BaseOptions.Delegate.GPU
    path_to_model : str = 'pose_landmarker_lite.task'
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    confidence_threshold: float = 0.5
    enable_segmentation: bool = True

MEDIAPIPE_TRACKER_LITE_PRESET = MediapipeGPUDetectorConfig(
    processor_type = python.BaseOptions.Delegate.GPU,
    path_to_model  = 'pose_landmarker_lite.task',
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=False
)

MEDIAPIPE_TRACKER_LITE_CPU_PRESET = MediapipeGPUDetectorConfig(
    processor_type = python.BaseOptions.Delegate.CPU,
    path_to_model  = 'pose_landmarker_lite.task',
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=False
)


class MediapipeGPUDetector(BaseDetector):
    config: MediapipeGPUDetectorConfig
    detector: mp.tasks.vision.PoseLandmarker

    @classmethod
    def create(cls, config: MediapipeGPUDetectorConfig|None=None) -> "MediapipeGPUDetector":
        if config is None:
            config = MediapipeGPUDetectorConfig()
        logger.info("entered GPU Detector construct")
        base_options = python.BaseOptions(model_asset_path=config.path_to_model,delegate=config.processor_type)
        options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=config.enable_segmentation)
        logger.info("creating detector")
        detector = vision.PoseLandmarker.create_from_options(options)
        logger.info("Returning from construct")
        return cls(
            config=config,
            detector=detector,
        )
    @classmethod
    def create_realtime_preset(cls) -> "MediapipeGPUDetector":
        logger.info("entered preset creation")
        return cls.create(config=MEDIAPIPE_TRACKER_LITE_PRESET)


    def detect(self, frame_number: int, image: np.ndarray) -> MediapipeGPUObservation:
        # print("pre gpu detect")
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # print(time.monotonic_ns() // 1_000_000)

        frame_rgb = np.ascontiguousarray(frame_rgb)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = time.monotonic_ns() // 1_000_000
        # print(timestamp_ms)
        mediapipe_results: MediapipeResults = self.detector.detect_for_video(mp_image,timestamp_ms) #New API has different type than ndarray
        # print("post gpu detect")
        if(mediapipe_results !=None and mediapipe_results.pose_landmarks != None and len(mediapipe_results.pose_landmarks) !=0):
            #logger.trace("mediapipe seems valid")
            pass
            
        else:
            logger.error("mediapipe seems invalid")
            print(mediapipe_results !=None)
            print(mediapipe_results.pose_landmarks != None)
            print(len(mediapipe_results.pose_landmarks))
        return MediapipeGPUObservation.from_detection_results(frame_number=frame_number,
                                                          mediapipe_results=mediapipe_results,
                                                          image_size=(int(image.shape[0]), int(image.shape[1])),
                                                          include_segmentation_mask=self.config.enable_segmentation
                                                          )

