from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os
import json
from datetime import timedelta



def detect_scene_changes(video_path, output_file, threshold=27.0):
    """
    用 PySceneDetect 分析视频分段
    threshold: 越低越敏感，越高越不敏感（一般 27~35）
    """
    print(f"正在检测场景切换: {video_path} (threshold={threshold})")

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor(1)
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    # 生成 JSON 格式
    scene_segments = []
    for start_time, end_time in scene_list:
        start_s = start_time.get_seconds()
        end_s = end_time.get_seconds()
        scene_segments.append({
            "start": str(timedelta(seconds=start_s)),
            "end": str(timedelta(seconds=end_s))
        })

    # 创建目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # 写 JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scene_segments, f, indent=2, ensure_ascii=False)

    print(f"\n场景切换检测完成，共生成 {len(scene_segments)} 个片段：")
    for i, seg in enumerate(scene_segments):
        print(f"片段 {i + 1}: {seg['start']} --> {seg['end']}")
