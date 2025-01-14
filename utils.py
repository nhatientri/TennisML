from scenedetect import open_video
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


def scene_detect(path_video):
    """
    Split video to disjoint fragments based on color histograms
    """
    video = open_video(path_video)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())

    scene_manager.detect_scenes(frame_source=video)
    scene_list = scene_manager.get_scene_list()

    # Handle case where no scenes are detected
    if not scene_list:
        # Treat the whole video as a single scene
        scene_list = [(video.base_timecode, video.duration)]
    scenes = [[x[0].frame_num, x[1].frame_num] for x in scene_list]
    return scenes
