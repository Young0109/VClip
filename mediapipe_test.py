import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def reframe_video(input_path: str, output_path: str, target_aspect_ratio=(9, 16), smoothing_factor=0.1, lost_target_threshold=15):
    """对输入的视频进行智能重构，并引入容错缓冲期来防止镜头跳动。"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算目标尺寸
    target_w_ratio, target_h_ratio = target_aspect_ratio
    target_width = original_width
    target_height = int(target_width * target_h_ratio / target_w_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    print(f"开始智能重构视频 (高级平滑模式): {input_path}")

    # --- 【关键改动】引入更完整的状态管理 ---
    # 上一帧的镜头裁剪位置
    last_known_crop_x = None
    # 目标丢失后的帧数计数器
    frames_since_lost = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        num_faces = 0
        if results.detections:
            num_faces = len(results.detections)

        # --- 【关键改动】引入容错缓冲逻辑 ---
        if num_faces > 0:
            # 只要找到脸，就重置丢失计数器
            frames_since_lost = 0
            
            # 计算当前帧最理想的镜头位置
            # 为了简化，我们以第一个检测到的人脸为主要目标
            target_crop_x = calculate_target_crop_x(frame, results.detections[0], target_width, target_height)
            
            # 平滑地更新镜头位置
            if last_known_crop_x is None:
                last_known_crop_x = target_crop_x
            else:
                last_known_crop_x = int(last_known_crop_x * (1.0 - smoothing_factor) + target_crop_x * smoothing_factor)

        else: # 如果当前帧没有检测到人脸
            frames_since_lost += 1
            
            # 如果目标丢失的时间没有超过阈值，镜头保持在原地不动
            if frames_since_lost > lost_target_threshold:
                # 只有当目标丢失太久时，才重置镜头位置（或切换到全景）
                last_known_crop_x = None

        # --- 根据最终的镜头位置生成画面 ---
        if last_known_crop_x is not None:
            # 如果有有效的目标位置，就进行裁剪
            new_frame = crop_frame_at(frame, last_known_crop_x, target_width, target_height)
        else:
            # 如果没有目标，就显示背景虚化的全景画面
            new_frame = create_blurred_background_frame(frame, target_width, target_height)

        out.write(new_frame)

    print(f"重构完成，已保存至: {output_path}")
    cap.release()
    out.release()

# --- 辅助函数 ---

def calculate_target_crop_x(frame, detection, target_w, target_h):
    """计算单帧最理想的裁剪位置X坐标"""
    h, w, _ = frame.shape
    box_data = detection.location_data.relative_bounding_box
    center_x = box_data.xmin + box_data.width / 2
    crop_w = int(h * target_w / target_h)
    return int((center_x * w) - (crop_w / 2))

def crop_frame_at(frame, crop_x, target_w, target_h):
    """在指定的X坐标处裁剪并缩放画面"""
    h, w, _ = frame.shape
    crop_w = int(h * target_w / target_h)
    
    # 边界检查
    crop_x = max(0, min(crop_x, w - crop_w))
    
    cropped_frame = frame[:, crop_x:crop_x+crop_w]
    return cv2.resize(cropped_frame, (target_w, target_h))

def create_blurred_background_frame(frame, target_w, target_h):
    """创建背景虚化的全景画面"""
    bg = cv2.resize(frame, (target_w, target_h))
    bg = cv2.GaussianBlur(bg, (99, 99), 0)
    h, w, _ = frame.shape
    scale = target_w / w
    fg_h, fg_w = int(h * scale), target_w
    fg = cv2.resize(frame, (fg_w, fg_h))
    y_offset = (target_h - fg_h) // 2
    bg[y_offset:y_offset+fg_h, 0:target_w] = fg
    return bg



    # =================================================================
#     以下是用于在本地电脑上独立测试本脚本的代码
# =================================================================

if __name__ == '__main__':
    
    full_input_path = "/Users/youngzhang/Desktop/highlight_005.mp4"
    
    # 输出文件将保存在与输入文件相同的目录中

    # --- 2. 构造完整的输入和输出路径 ---
    
    # 从输入路径中获取目录名
    input_directory = os.path.dirname(full_input_path)
    
    # 根据输入路径自动生成输出文件的路径
    base_name = os.path.basename(full_input_path)
    file_name, file_ext = os.path.splitext(base_name)
    # 将输出文件保存在与输入文件相同的目录中
    full_output_path = os.path.join(input_directory, f"{file_name}_reframed_9x16{file_ext}")

    # --- 3. 检查输入文件并执行重构任务 ---

    # 检查您提供的视频路径是否存在
    if not os.path.exists(full_input_path):
        print(f"\n❌ 错误：找不到您提供的视频文件！")
        print(f"请检查路径是否正确: {full_input_path}")
        # 如果文件不存在，直接退出，避免报错
        exit()
    
    print(f"\n[测试任务开始]")
    print(f"  - 输入视频: {full_input_path}")
    print(f"  - 输出路径: {full_output_path}")

    # 调用核心功能函数，传入本地路径
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
