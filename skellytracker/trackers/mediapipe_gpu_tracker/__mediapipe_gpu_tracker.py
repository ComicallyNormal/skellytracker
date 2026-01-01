import logging

from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker, BaseTrackerConfig, BaseRecorder
from skellytracker.trackers.mediapipe_gpu_tracker.mediapipe_gpu_annotator import MediapipeGPUAnnotatorConfig, \
    MediapipeGPUImageAnnotator
from skellytracker.trackers.mediapipe_gpu_tracker.mediapipe_gpu_detector import MediapipeGPUDetector, MediapipeGPUDetectorConfig

logger = logging.getLogger(__name__)

class MediapipeGPUTrackerConfig(BaseTrackerConfig):
    detector_config: MediapipeGPUDetectorConfig = Field(default_factory = MediapipeGPUDetectorConfig)
    annotator_config: MediapipeGPUAnnotatorConfig = Field(default_factory = MediapipeGPUAnnotatorConfig)

class MediapipeRecorder(BaseRecorder):
    # TODO: the BaseRecorder covers most of this, but we could save metadata with this if we wanted
    pass


class MediapipeGPUTracker(BaseTracker):
    config: MediapipeGPUTrackerConfig
    detector: MediapipeGPUDetector
    annotator: MediapipeGPUImageAnnotator | None = None
    recorder: MediapipeRecorder | None = None

    @classmethod
    def create(cls, config: MediapipeGPUTrackerConfig | None = None):
        if config is None:
            config = MediapipeGPUTrackerConfig()
        detector = MediapipeGPUDetector.create(config.detector_config)

        return cls(
            config=config,
            detector=detector,
            annotator=MediapipeGPUImageAnnotator.create(config.annotator_config),
            recorder=MediapipeRecorder(),
        )


if __name__ == "__main__":
    MediapipeGPUTracker.create().demo()
