import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def reframe_video(input_path: str, output_path: str, target_aspect_ratio=(9, 16), smoothing_factor=0.1, lost_target_threshold=15):
    """对输入的视频进行智能重构，并使用 FFmpeg 安全地合并原始音频。"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    target_w_ratio, target_h_ratio = target_aspect_ratio
    target_width = original_width
    target_height = int(target_width * target_h_ratio / target_w_ratio)

    temp_silent_output_path = output_path.replace('.mp4', '_silent_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_silent_output_path, fourcc, fps, (target_width, target_height))

    print(f"步骤 1/2: 开始智能重构视频画面: {input_path}")

    last_known_crop_x = None
    frames_since_lost = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        num_faces = 0
        if results.detections:
            num_faces = len(results.detections)

        # --- 容错缓冲逻辑 ---
        if num_faces > 0:
            frames_since_lost = 0
            target_crop_x = calculate_target_crop_x(frame, results.detections[0], target_width, target_height)
            
            if last_known_crop_x is None:
                last_known_crop_x = target_crop_x
            else:
                last_known_crop_x = int(last_known_crop_x * (1.0 - smoothing_factor) + target_crop_x * smoothing_factor)
        else:
            frames_since_lost += 1
            if frames_since_lost > lost_target_threshold:
                last_known_crop_x = None

        # --- 根据最终的镜头位置生成画面 ---
        if last_known_crop_x is not None:
            new_frame = crop_frame_at(frame, last_known_crop_x, target_width, target_height)
        else:
            new_frame = create_blurred_background_frame(frame, target_width, target_height)

        out.write(new_frame)

    print("画面处理完成。")
    cap.release()
    out.release()

    # --- 【关键改动】步骤 2: 使用 FFmpeg 合并音频 ---
    print(f"步骤 2/2: 正在使用 FFmpeg 合并原始音频...")
    try:
        # 构造 FFmpeg 命令
        # -i [无声视频] -i [原始视频] -c:v copy (直接复制视频流，不重新编码，速度极快)
        # -map 0:v:0 (使用第一个输入的视频流) -map 1:a:0 (使用第二个输入的音频流)
        # -shortest (以最短的流为准结束)
        command = [
            'ffmpeg',
            '-y',  # 无需询问，直接覆盖输出文件
            '-i', temp_silent_output_path,
            '-i', input_path,
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
        
        # 执行命令
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        print(f"音频合并成功！最终文件已保存至: {output_path}")

    except FileNotFoundError:
        print("错误：找不到 'ffmpeg' 命令。请确保您已成功安装 FFmpeg。")
        print(f"无声视频已保留在: {temp_silent_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"错误：FFmpeg 在执行时出错。")
        print(f"FFmpeg 错误信息: {e.stderr}")
        print(f"无声视频已保留在: {temp_silent_output_path}")
    finally:
        if os.path.exists(temp_silent_output_path):
            os.remove(temp_silent_output_path)


def calculate_target_crop_x(frame, detection, target_w, target_h):
    h, w, _ = frame.shape
    box_data = detection.location_data.relative_bounding_box
    center_x = box_data.xmin + box_data.width / 2
    crop_w = int(h * target_w / target_h)
    return int((center_x * w) - (crop_w / 2))

def crop_frame_at(frame, crop_x, target_w, target_h):
    h, w, _ = frame.shape
    crop_w = int(h * target_w / target_h)
    crop_x = max(0, min(crop_x, w - crop_w))
    cropped_frame = frame[:, crop_x:crop_x+crop_w]
    return cv2.resize(cropped_frame, (target_w, target_h))

def create_blurred_background_frame(frame, target_w, target_h):
    bg = cv2.resize(frame, (target_w, target_h))
    bg = cv2.GaussianBlur(bg, (99, 99), 0)
    h, w, _ = frame.shape
    scale = target_w / w
    fg_h, fg_w = int(h * scale), target_w
    fg = cv2.resize(frame, (fg_w, fg_h))
    y_offset = (target_h - fg_h) // 2
    bg[y_offset:y_offset+fg_h, 0:target_w] = fg
    return bg

if __name__ == '__main__':
    
    full_input_path = ("/Users/youngzhang/Desktop/highlight_006"
                       ".mp4")
    
    input_directory = os.path.dirname(full_input_path)
    
    base_name = os.path.basename(full_input_path)
    file_name, file_ext = os.path.splitext(base_name)
    full_output_path = os.path.join(input_directory, f"{file_name}_reframed_9x16{file_ext}")

    if not os.path.exists(full_input_path):
        print(f"\n❌ 错误：找不到您提供的视频文件！")
        print(f"请检查路径是否正确: {full_input_path}")
        # 如果文件不存在，直接退出，避免报错
        exit()
    
    print(f"\n[测试任务开始]")
    print(f"  - 输入视频: {full_input_path}")
    print(f"  - 输出路径: {full_output_path}")

    reframe_video(
        input_path=full_input_path, 
        output_path=full_output_path, 
        target_aspect_ratio=(9, 16),
        smoothing_factor=0.08,
        lost_target_threshold=15 
    )

    print(f"\n[测试任务完成]")
    print(f"请在输入视频所在的文件夹中查看重构后的视频: {full_output_path}")
    print("--- 独立测试模式运行结束 ---")