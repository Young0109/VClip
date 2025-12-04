import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import json
import time
import requests
import statistics
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

def get_scene_cuts(video_path: str, threshold: float = 27.0) -> list:
    """【场景切换检测器】使用 PySceneDetect 库来精确地检测视频中的所有硬切时刻。
    返回一个包含所有场景切换时间点（秒）的列表。
    """
    print(f"\n场景分析阶段: 开始使用 PySceneDetect 检测硬切点 -> {video_path}")
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video, show_progress=True)
        cut_list = scene_manager.get_cut_list()
        cut_times_seconds = [cut.get_seconds() for cut in cut_list]
        if cut_times_seconds:
             print(f"场景分析阶段: 检测到 {len(cut_times_seconds)} 个硬切点。")
        else:
             print("场景分析阶段: 未检测到明显的硬切点。")
        return cut_times_seconds
    except Exception as e:
        print(f"错误: PySceneDetect 分析失败: {e}。将返回空列表。")
        return []

# -----------------------------------------------------------------------------
# SECTION 1: 初始化与配置
# -----------------------------------------------------------------------------

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

# -----------------------------------------------------------------------------
# SECTION 2: 核心工作流函数 (分析 -> 平滑 -> 决策 -> 后处理 -> 执行)
# -----------------------------------------------------------------------------

def analyze_video_scenes(input_path: str, analysis_interval, main_face_area_threshold):
    print(f"分析阶段: 开始快速分析视频场景 (间隔: {analysis_interval}s) -> {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("错误: 无法打开视频。")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("错误: 无法读取视频的FPS，将使用默认值30。")
        fps = 30

    scenes = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps

        if frame_count % int(fps * analysis_interval) == 0:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = results.detections if results.detections else []

            face_areas = [(d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height) for d in detections]
            main_face_count = sum(1 for area in face_areas if area > main_face_area_threshold)
            max_area = max(face_areas) if face_areas else 0.0

            scene_info = {
                "time": round(current_time, 2),
                "total_face_count": len(detections),
                "main_face_count": main_face_count,
                "max_face_area": round(max_area, 4)
            }
            scenes.append(scene_info)
            
            print(f"\r分析阶段: 进度 {frame_count}/{total_frames} ({current_time:.2f}s)", end="")

        frame_count += 1

    cap.release()
    print("\n分析阶段: 完成！")
    return scenes

def smooth_scene_report(scenes: list, window_seconds: float):
    """【V2版-平滑处理器】"""
    print(f"\n平滑处理阶段 (v2): 开始优化报告，优先保留特写 (窗口: {window_seconds}s)...")
    if not scenes:
        return []

    smoothed_scenes = json.loads(json.dumps(scenes))
    num_scenes = len(scenes)

    for i in range(num_scenes):
        current_time = scenes[i]['time']
        window_start_time = current_time - window_seconds / 2
        window_end_time = current_time + window_seconds / 2

        protagonist_was_present_in_window = False
        for j in range(num_scenes):
            if window_start_time <= scenes[j]['time'] <= window_end_time:
                if scenes[j]['main_face_count'] >= 1:
                    protagonist_was_present_in_window = True
                    break

        if protagonist_was_present_in_window and smoothed_scenes[i]['main_face_count'] == 0:
            smoothed_scenes[i]['main_face_count'] = 1

    print("平滑处理阶段 (v2): 完成！")
    return smoothed_scenes

def stabilize_protagonist_report_v3(scenes: list, grace_period: float, scene_cuts: list):
    """【V3版-决策稳定器】"""
    print(f"\n决策稳定阶段 (v3): 开始稳定主角判定 (宽限期: {grace_period}s, 感知硬切)...")
    if not scenes:
        return []

    stabilized_scenes = json.loads(json.dumps(scenes))
    
    last_protagonist_seen_time = -grace_period
    cut_index = 0

    for scene in stabilized_scenes:
        current_time = scene['time']
        
        if cut_index < len(scene_cuts) and current_time >= scene_cuts[cut_index]:
            print(f"  - 硬切点处理: 时间点 {current_time:.2f}s, 遇到场景切换，重置宽限期状态！")
            last_protagonist_seen_time = -grace_period
            cut_index += 1

        if scene['main_face_count'] >= 1:
            last_protagonist_seen_time = current_time
        else:
            time_since_last_seen = current_time - last_protagonist_seen_time
            if time_since_last_seen < grace_period:
                scene['main_face_count'] = 1

    print("决策稳定阶段 (v3): 完成！")
    return stabilized_scenes

def stabilize_multi_protagonist_report(scenes: list, commitment_period: float) -> list:
    """【多主角决策稳定器】解决在1,2人之间摇摆不定导致的“分屏抖动”问题。
    一旦检测到2个或以上的主角，就在接下来的 'commitment_period'  内，
    即使暂时只检测到1个人，也强制将其报告为2人，以锁定分屏决策。
    """
    print(f"\n多主角稳定阶段: 开始锁定分屏决策 (承诺期: {commitment_period}s)...")
    if not scenes:
        return []

    stabilized_scenes = json.loads(json.dumps(scenes))
    
    last_two_protagonists_seen_time = -commitment_period

    for scene in stabilized_scenes:
        current_time = scene['time']
        
        if scene['main_face_count'] >= 2:
            last_two_protagonists_seen_time = current_time
        
        elif scene['main_face_count'] == 1:
            time_since_last_seen_two = current_time - last_two_protagonists_seen_time
            if time_since_last_seen_two < commitment_period:
                print(f"  - 稳定修正: 时间点 {current_time:.2f}s, 仍在分屏承诺期内。将主角数 1 强制修正为 2。")
                scene['main_face_count'] = 2
                
    print("多主角稳定阶段: 完成！")
    return stabilized_scenes

def add_context_to_report(scenes: list) -> list:
    """【趋势分析器】为场景报告添加上下文信息，帮助AI理解主角数量的变化趋势。"""
    if not scenes:
        return []
    
    contextual_scenes = json.loads(json.dumps(scenes))
    
    for i in range(len(contextual_scenes)):
        current_faces = contextual_scenes[i]['main_face_count']
        prev_faces = contextual_scenes[i-1]['main_face_count'] if i > 0 else -1
        
        trend = "stable"
        
        if current_faces > 0 and prev_faces == 0:
            trend = "appearing"
        elif current_faces == 0 and prev_faces > 0:
            trend = "fading"
        elif current_faces > prev_faces and prev_faces != -1:
            trend = "increasing"
        elif current_faces < prev_faces:
             trend = "decreasing"

        contextual_scenes[i]['protagonist_trend'] = trend
        
    print("\n趋势分析阶段: 已为报告添加主角变化趋势信息。")
    return contextual_scenes

def get_edit_plan_from_qwen(scenes_report: list):
    print("\n决策阶段: 正在生成趋势感知型Prompt并请求Qwen API...")

    prompt_text = (
        "你是一位顶级的短视频剪辑师，擅长制作快节奏、抓人眼球的竖屏内容。你的核心任务是**运镜**，而不是简单的切镜。\n"
        "请根据以下包含**主角趋势**的视频分析报告，为我制定一个9:16的竖屏剪辑计划。\n\n"
        "【剪辑规则】:\n"
        "1.  **单人特写 (CLOSE_UP):** 当'主角脸数量'为 1 时首选。使用平滑摇移（Pan）保持人物居中。\n"
        "2.  **双人分屏 (SPLIT_SCREEN):** 当'主角脸数量'稳定为 2 时使用。一旦决定使用分屏，请尽量保持，不要因为数量短暂变为1就立即切走。\n"
        "3.  **远景/全景 (WIDE_SHOT_BLUR):**\n"
        "    - 当'主角脸数量'为 0 时使用，用来交代环境或展示多人场景。\n"
        "    - **关键：当'主角趋势'显示为 'fading' (消失)时，你应该立即、果断地切换到 WIDE_SHOT，不要犹豫！**\n"
        "4.  **【核心原则】避免生硬切换和反弹:**\n"
        "    - **连贯性:** 尽量保持一个镜头类型持续一个有意义的时长（例如至少1.5-2秒）。\n"
        "    - **决策惯性:** 当你从CLOSE_UP切换到WIDE_SHOT后，即使'主角趋势'短暂变为'appearing'，也请保持WIDE_SHOT至少1.5秒，以观察情况，避免立即切回造成的“反弹”。\n\n"
        "【视频分析报告】:\n"
    )

    for scene in scenes_report:
        prompt_text += (
            f"- 时间点 {scene['time']}秒: 有 {scene['main_face_count']} 个主角脸 "
            f"(趋势: {scene.get('protagonist_trend', 'stable')}), "
            f"最大脸面积占画面的 {scene['max_face_area']:.2%}。\n"
        )

    prompt_text += "\n请严格按照以下JSON格式返回你的剪辑计划 (Edit Decision List)，不要包含任何JSON格式之外的额外说明文字:\n{\"edit_plan\": [{\"start_time\": float, \"end_time\": float, \"shot_type\": \"SHOT_TYPE\", \"description\": \"你的剪辑理由(请简洁)\"}]}"
    
    api_key = "sk-0a0eefabc3f9421399d0f5981904326b" 
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "qwen-max", "input": {"prompt": prompt_text}, "parameters": {}}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        qwen_response_json = response.json()
        generated_text = ""
        if "output" in qwen_response_json and "text" in qwen_response_json["output"]:
            generated_text = qwen_response_json["output"]["text"]
            print("决策阶段: 成功获取Qwen的剪辑建议！")
            
            json_part = generated_text
            if "```json" in json_part:
                json_part = json_part.split("```json")[1].split("```")[0]
            
            start_index = json_part.find('{')
            end_index = json_part.rfind('}')
            if start_index != -1 and end_index != -1:
                json_part = json_part[start_index:end_index+1]
                return json.loads(json_part)
            else:
                 print(f"错误: 在Qwen返回的内容中找不到有效的JSON对象。")
                 return None
        else:
            print(f"错误: Qwen返回的数据格式不正确。返回内容: {qwen_response_json}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"错误: API请求失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 解析Qwen返回的JSON失败: {e}. 返回的文本内容是: \n{generated_text}")
        return None


def refine_edit_plan_v2(edit_plan: dict, min_duration: float, commit_duration: float):
    """解决反弹问题"""

    print(f"\n决策后处理阶段 (v2): 开始用'决策惯性'优化剪辑计划 (最短时长: {min_duration}s, 承诺时长: {commit_duration}s)...")
    original_plan = edit_plan.get("edit_plan")
    if not original_plan or len(original_plan) < 2:
        print("决策后处理阶段: 剪辑计划过短，无需优化。")
        return edit_plan

    refined_plan = []
    i = 0
    while i < len(original_plan):
        current_clip = original_plan[i]
        
        if (i + 2 < len(original_plan) and
                original_plan[i+1]['shot_type'] != current_clip['shot_type'] and
                original_plan[i+2]['shot_type'] == current_clip['shot_type']):
            
            next_clip = original_plan[i+1]
            flicker_duration = next_clip['end_time'] - next_clip['start_time']
            
            if flicker_duration < commit_duration:
                print(f"  - 修复反弹: 发现不稳定的切换 ({current_clip['shot_type']} -> {next_clip['shot_type']} -> {original_plan[i+2]['shot_type']})。")
                print(f"    -> 中间片段时长 {flicker_duration:.2f}s 过短，强制合并。")
                
                merged_clip = {
                    "start_time": current_clip['start_time'],
                    "end_time": original_plan[i+2]['end_time'],
                    "shot_type": current_clip['shot_type'],
                    "description": f"合并了不稳定的切换 ({current_clip['description']} & {original_plan[i+2]['description']})"
                }
                refined_plan.append(merged_clip)
                i += 3
                continue

        duration = current_clip["end_time"] - current_clip["start_time"]
        if duration < min_duration and i + 1 < len(original_plan) and current_clip["shot_type"] == original_plan[i+1]["shot_type"]:
            print(f"  - 合并短片段: 时长 {duration:.2f}s 过短，与下一个同类型片段合并。")
            next_clip = original_plan[i+1]
            merged_clip = {
                "start_time": current_clip['start_time'],
                "end_time": next_clip['end_time'],
                "shot_type": current_clip['shot_type'],
                "description": f"{current_clip['description']} & {next_clip['description']}"
            }
            refined_plan.append(merged_clip)
            i += 2
            continue
            
        refined_plan.append(current_clip)
        i += 1
    
    if not refined_plan:
        print("决策后处理阶段: 优化后计划为空，返回原始计划。")
        return edit_plan
        
    for j in range(len(refined_plan) - 1):
        refined_plan[j]['end_time'] = refined_plan[j+1]['start_time']
    
    if refined_plan and original_plan:
        refined_plan[-1]['end_time'] = original_plan[-1]['end_time']
    
    edit_plan["edit_plan"] = refined_plan
    print("决策后处理阶段 (v2): 完成！")
    return edit_plan


def execute_edit_plan(input_path: str, output_path: str, edit_plan: dict, smoothing_factor: float):
    """解决跳动问题"""

    print("\n执行阶段 (v3): 开始按照最终剪辑计划处理视频 ...")

    plan = edit_plan.get("edit_plan")
    if not plan:
        print("错误: 剪辑计划无效。")
        return

    cap = cv2.VideoCapture(input_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    target_aspect_ratio = (9, 16)
    target_w_ratio, target_h_ratio = target_aspect_ratio
    
    target_width = 1080
    target_height = int(target_width * target_h_ratio / target_w_ratio)

    if target_width % 2 != 0: target_width -= 1
    if target_height % 2 != 0: target_height -= 1
    
    temp_silent_output_path = output_path.replace('.mp4', '_silent_temp.mp4')

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{target_width}x{target_height}',  
        '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        temp_silent_output_path
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    plan_index = 0
    
    last_known_crop_x_single = None
    last_known_crop_x_split1 = None
    last_known_crop_x_split2 = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps
        print(f"\r执行阶段: 正在处理第 {frame_count}/{total_frames} 帧 ({current_time:.2f}s)", end="")

        if plan_index + 1 < len(plan) and current_time >= plan[plan_index + 1]["start_time"]:
            plan_index += 1
            print(f"\n  - 时间点 {current_time:.2f}s: 切换片段 -> {plan[plan_index]['shot_type']} (理由: {plan[plan_index]['description']})")
            last_known_crop_x_single = None
            last_known_crop_x_split1 = None
            last_known_crop_x_split2 = None

        current_shot_type = plan[plan_index]["shot_type"]
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = results.detections if results.detections else []
        
        if current_shot_type == "WIDE_SHOT_BLUR":
            new_frame = create_blurred_background_frame(frame, target_width, target_height)
        
        elif current_shot_type == "CLOSE_UP":
            if detections:
                main_subject = sorted(detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, reverse=True)[0]
                new_frame, last_known_crop_x_single = create_pan_and_scan_frame(frame, main_subject, target_width, target_height, last_known_crop_x_single, smoothing_factor)
            else:
                if last_known_crop_x_single is not None:
                    new_frame = crop_frame_at(frame, last_known_crop_x_single, target_width, target_height)
                else:
                    new_frame = create_blurred_background_frame(frame, target_width, target_height)
        
        elif current_shot_type == "SPLIT_SCREEN":
            top_two = get_top_two_faces(detections)
            current_x1, current_x2 = None, None

            if len(top_two) >= 2:
                target1_x = calculate_target_crop_x(frame, top_two[0], target_width, target_height // 2)
                target2_x = calculate_target_crop_x(frame, top_two[1], target_width, target_height // 2)

                if last_known_crop_x_split1 is not None and abs(target1_x - last_known_crop_x_split1) > abs(target2_x - last_known_crop_x_split1):
                    target1_x, target2_x = target2_x, target1_x
                
                if target1_x > target2_x:
                     target1_x, target2_x = target2_x, target1_x

                current_x1 = target1_x
                current_x2 = target2_x

            elif len(top_two) == 1:
                target_x = calculate_target_crop_x(frame, top_two[0], target_width, target_height // 2)
                
                if last_known_crop_x_split1 is None or \
                   abs(target_x - last_known_crop_x_split1) < abs(target_x - (last_known_crop_x_split2 or (original_width*2))):
                    current_x1 = target_x
                    current_x2 = last_known_crop_x_split2
                else:
                    current_x1 = last_known_crop_x_split1
                    current_x2 = target_x
            else:
                current_x1 = last_known_crop_x_split1
                current_x2 = last_known_crop_x_split2

            new_x1 = int(last_known_crop_x_split1 * (1.0 - smoothing_factor) + current_x1 * smoothing_factor) if last_known_crop_x_split1 is not None and current_x1 is not None else current_x1
            new_x2 = int(last_known_crop_x_split2 * (1.0 - smoothing_factor) + current_x2 * smoothing_factor) if last_known_crop_x_split2 is not None and current_x2 is not None else current_x2

            last_known_crop_x_split1 = new_x1 if new_x1 is not None else last_known_crop_x_split1
            last_known_crop_x_split2 = new_x2 if new_x2 is not None else last_known_crop_x_split2
            
            frame1 = crop_frame_at(frame, last_known_crop_x_split1, target_width, target_height // 2)
            frame2 = crop_frame_at(frame, last_known_crop_x_split2, target_width, target_height // 2)
            new_frame = np.vstack((frame1, frame2))

        else:
            new_frame = create_blurred_background_frame(frame, target_width, target_height)
        
        try:
            process.stdin.write(new_frame.tobytes())
        except (IOError, BrokenPipeError) as e:
            print(f"\n错误：写入 FFmpeg 管道失败: {e}")
            break
        
        frame_count += 1

    print("\n执行阶段: 画面处理完成，正在关闭FFmpeg管道...")
    cap.release()
    process.stdin.close()
    stderr = process.stderr.read().decode('utf-8', errors='ignore')
    process.wait()

    if process.returncode != 0:
        print(f"❌ 错误：FFmpeg 在处理视频帧时返回了错误。")
        print(f"FFmpeg Stderr:\n{stderr}")
    
    print("\n最后一步: 正在合并音频...")
    if os.path.exists(temp_silent_output_path) and os.path.getsize(temp_silent_output_path) > 0:
        merge_audio(temp_silent_output_path, input_path, output_path)
    else:
        print(f"❌ 错误：无声视频文件 '{temp_silent_output_path}' 未能成功生成，跳过音频合并。")

# -----------------------------------------------------------------------------
# SECTION 4: 辅助与渲染函数
# -----------------------------------------------------------------------------
def merge_audio(silent_video_path, original_video_path, final_output_path):

    try:
        command = ['ffmpeg', '-y', '-i', silent_video_path, '-i', original_video_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', final_output_path]
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ 成功！最终文件已保存至: {final_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误：音频合并失败。FFmpeg返回了错误。")
        print(f"FFmpeg Stderr: {e.stderr}")
    except Exception as e:
        print(f"❌ 错误：音频合并失败。错误: {e}")
    finally:
        if os.path.exists(silent_video_path):
            os.remove(silent_video_path)

def get_top_two_faces(detections):

    if not detections:
        return []
    detections.sort(key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, reverse=True)
    return detections[:2]


def calculate_target_crop_x(frame, detection, target_w, target_h):

    h, w, _ = frame.shape
    box_data = detection.location_data.relative_bounding_box
    center_x = box_data.xmin + box_data.width / 2
    crop_w_in_original = int(h * target_w / target_h)
    return int((center_x * w) - (crop_w_in_original / 2))

def crop_frame_at(frame, crop_x, target_w, target_h):

    h, w, _ = frame.shape
    crop_w_in_original = int(h * target_w / target_h)
    crop_x = max(0, min(crop_x, w - crop_w_in_original)) if crop_x is not None else (w - crop_w_in_original) // 2
    cropped_frame = frame[:, crop_x:crop_x + crop_w_in_original]
    return cv2.resize(cropped_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

def create_blurred_background_frame(frame, target_w, target_h):

    bg = cv2.resize(frame, (target_w, target_h))
    bg = cv2.GaussianBlur(bg, (155, 155), 0)
    h, w, _ = frame.shape
    scale = target_w / w
    fg_h, fg_w = int(h * scale), target_w
    fg = cv2.resize(frame, (fg_w, fg_h))
    y_offset = (target_h - fg_h) // 2
    bg[y_offset:y_offset+fg_h, 0:target_w] = fg
    return bg


def create_pan_and_scan_frame(frame, detection, target_w, target_h, previous_crop_x, smoothing_factor):

    target_crop_x = calculate_target_crop_x(frame, detection, target_w, target_h)
    new_crop_x = int(previous_crop_x * (1.0 - smoothing_factor) + target_crop_x * smoothing_factor) if previous_crop_x is not None else target_crop_x
    final_frame = crop_frame_at(frame, new_crop_x, target_w, target_h)
    return final_frame, new_crop_x


def create_split_screen_frame(frame, detections, target_w, target_h, prev_x1, prev_x2, smoothing_factor):

    if not detections:
        if prev_x1 is not None and prev_x2 is not None:
            frame1 = crop_frame_at(frame, prev_x1, target_w, target_h // 2)
            frame2 = crop_frame_at(frame, prev_x2, target_w, target_h // 2)
            return np.vstack((frame1, frame2)), prev_x1, prev_x2
        else:
            return None, None, None

    top_two = get_top_two_faces(detections)
    if len(top_two) < 2: 
        return None, None, None

    target1_x = calculate_target_crop_x(frame, top_two[0], target_w, target_h // 2)
    target2_x = calculate_target_crop_x(frame, top_two[1], target_w, target_h // 2)

    if prev_x1 is not None and prev_x2 is not None:
        if abs(target1_x - prev_x1) + abs(target2_x - prev_x2) > abs(target1_x - prev_x2) + abs(target2_x - prev_x1):
            target1_x, target2_x = target2_x, target1_x

    if target1_x > target2_x:
        target1_x, target2_x = target2_x, target1_x

    new_x1 = int(prev_x1 * (1.0 - smoothing_factor) + target1_x * smoothing_factor) if prev_x1 is not None else target1_x
    new_x2 = int(prev_x2 * (1.0 - smoothing_factor) + target2_x * smoothing_factor) if prev_x2 is not None else target2_x

    frame1 = crop_frame_at(frame, new_x1, target_w, target_h // 2)
    frame2 = crop_frame_at(frame, new_x2, target_w, target_h // 2)

    return np.vstack((frame1, frame2)), new_x1, new_x2

    
def reframe_to_vertical_video(input_video_path: str, output_video_path: str) -> str or None:

    print(f"=== Qwen驱动的全自动智能剪辑任务开始 ===")
    print(f"输入文件: {input_video_path}")
    print(f"输出文件: {output_video_path}")
    
    # --- 工作流核心参数定义 ---
    ANALYSIS_INTERVAL = 0.25
    MAIN_FACE_THRESHOLD = 0.006
    SMOOTHING_WINDOW = 1.5
    MIN_CLIP_DURATION = 1.0
    CAMERA_SMOOTHING_FACTOR = 0.05 
    SCENE_DETECT_THRESHOLD = 27.0
    
    # --- 针对决策“抖动”问题的核心参数 ---
    GRACE_PERIOD_FOR_WIDE_SHOT = 1.0
    COMMITMENT_DURATION = 1.5
    SPLIT_SCREEN_COMMITMENT_PERIOD = 2.0

    # --- 执行优化后的工作流 ---
    # 1. 场景切换检测
    scene_cuts = get_scene_cuts(input_video_path, threshold=SCENE_DETECT_THRESHOLD)

    # 2. 分析视频
    raw_scenes_report = analyze_video_scenes(input_video_path, 
                                             analysis_interval=ANALYSIS_INTERVAL, 
                                             main_face_area_threshold=MAIN_FACE_THRESHOLD)
    
    if not raw_scenes_report:
        print("错误: 无法生成初始场景报告，智能转换终止。")
        return None
        
    # 3. 对报告进行平滑处理 (解决 0/1 抖动)
    smoothed_report = smooth_scene_report(raw_scenes_report, window_seconds=SMOOTHING_WINDOW)

    # 4. 稳定主角决策 (解决短暂消失问题)
    stabilized_report = stabilize_protagonist_report_v3(smoothed_report,
                                                        grace_period=GRACE_PERIOD_FOR_WIDE_SHOT,
                                                        scene_cuts=scene_cuts)
    
    # 5. 稳定分屏决策 (解决 1/2 抖动)
    final_stabilized_report = stabilize_multi_protagonist_report(stabilized_report, 
                                                                 commitment_period=SPLIT_SCREEN_COMMITMENT_PERIOD)

    # 6. 为报告添加趋势上下文
    contextual_report = add_context_to_report(final_stabilized_report)
                                                    
    # 7. 从Qwen获取剪辑计划
    edit_plan = get_edit_plan_from_qwen(contextual_report)
    
    if not edit_plan:
        print("错误: 无法从Qwen获取剪辑计划，智能转换终止。")
        return None

    # 8. 使用后处理器优化AI的原始计划
    refined_edit_plan = refine_edit_plan_v2(edit_plan, 
                                            min_duration=MIN_CLIP_DURATION, 
                                            commit_duration=COMMITMENT_DURATION)
        
    # 9. 严格按照最终的、精修过的计划执行剪辑
    execute_edit_plan(input_video_path, output_video_path, refined_edit_plan, smoothing_factor=CAMERA_SMOOTHING_FACTOR)

    print("\n[ Qwen驱动的全自动智能剪辑任务全部完成 ]")
