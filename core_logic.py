import os
import sys
import json
import requests
import shutil
from datetime import timedelta
import statistics

sys.path.append(os.path.abspath('Video & Audio Extraction'))
sys.path.append(os.path.abspath('Image Difference Detection'))
sys.path.append(os.path.abspath('Semantic Analysis Pipeline'))
sys.path.append(os.path.abspath('service'))

from videoExtract import convert_to_mp4
from videoExtract import extract_frames
from imgDifference import detect_scene_changes
from visual_scorer_api import run_visual_scoring_pipeline
from clip_export import export_final_clips 
from videoExtract import extract_audio
from vocal_separator import separate_vocals
from transcription import transcribe_audio
from segment_text import semantic_segment_final
from scoring_pipeline import run_scoring_pipeline
from intelligent_reframe import reframe_to_vertical_video 

def download_video(video_url: str, save_dir: str) -> str | None:
    """从URL下载视频，或者直接复制本地文件。"""
    
    # 【新增逻辑】判断是不是我们自己构造的本地文件 URL
    if video_url.startswith("file://"):
        local_path = video_url[7:] # 去掉前面的 file://
        print(f"[核心逻辑] 检测到本地文件路径: {local_path}")
        
        if not os.path.exists(local_path):
            print(f"错误: 本地文件不存在: {local_path}")
            return None
            
        try:
            filename = "input_video.original"
            save_path = os.path.join(save_dir, filename)
            shutil.copy2(local_path, save_path) # 直接复制过去
            print(f"本地视频已复制至工作目录: {save_path}")
            return save_path
        except Exception as e:
            print(f"错误: 复制本地文件失败: {e}")
            return None

    # --- 下面是原来的网络下载逻辑 (保持不变) ---
    try:
        print(f"[核心逻辑] 开始从 URL 下载视频: {video_url}")
        headers = {'User-Agent': 'Mozilla/5.0 ...'} # (原来的 User-Agent)
        
        # ... (保留你原来的下载代码) ...
        response = requests.get(video_url, stream=True, timeout=1800, headers=headers)
        response.raise_for_status()

        filename = "input_video.original"
        save_path = os.path.join(save_dir, filename)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"视频下载成功，保存至: {save_path}")
        return save_path
    except Exception as e:
        print(f" 错误：视频下载失败。URL: {video_url}, 错误: {e}")
        return None

def send_completion_callback(task_id: str, status: str, results: dict):
    """向FastAPI发送任务完成或失败的回调通知。"""
    callback_url = "http://127.0.0.1:8000/tasks/callback" # 【接口契约】
    payload = {"task_id": task_id, "status": status, "results": results}
    try:
        print(f"正在发送回调通知: task_id={task_id}, status={status}")
        requests.post(callback_url, json=payload, timeout=300)
        print(" 回调通知发送成功。")
    except requests.RequestException as e:
        print(f"错误：回调通知发送失败。错误: {e}")




def timedelta_to_seconds(time_str: str) -> float:
    """(修复版) 将 'HH:MM:SS.ms' 或 'H:MM:SS' 格式的时间字符串精确转换为浮点数秒。"""
    if not isinstance(time_str, str):
        return 0.0

    if '.' in time_str:
        main_part, ms_part = time_str.split('.')
        if not ms_part.isdigit():
            ms_part = "0"
    else:
        main_part, ms_part = time_str, "0"

    parts = main_part.split(':')
    if len(parts) == 3:
        try:
            h, m, s = map(int, parts)
            total_seconds = float(h * 3600 + m * 60 + s)
            total_seconds += float(f"0.{ms_part}")
            return total_seconds
        except ValueError:
            return 0.0 
    return 0.0

def fuse_scores(visual_scores_path, text_scores_path, output_path, w_visual=0.6, w_text=0.4):
    with open(visual_scores_path, 'r', encoding='utf-8') as f:
        visual_segments = json.load(f)
    with open(text_scores_path, 'r', encoding='utf-8') as f:
        text_segments = json.load(f)

    all_text_scores = [seg.get('final_text_score', 0) for seg in text_segments]
    if not all_text_scores:
        print("警告: 文本分数列表为空。")
        return

    max_text_score = max(all_text_scores)
    min_text_score = min(all_text_scores)
    score_range = max_text_score - min_text_score
    
    print(f"文本分数分析: 最高分={max_text_score}, 最低分={min_text_score}, 分数范围={score_range}")

    if score_range == 0:
        print("警告: 所有文本分数都相同，将所有文本归一化分数设为0.5。")

    combined_results = []
    
    for v_seg in visual_segments:
        v_start_sec = timedelta_to_seconds(v_seg['start'])
        v_end_sec = timedelta_to_seconds(v_seg['end'])
        v_score = v_seg.get('total_visual_score', 0)

        overlapping_raw_scores = []
        overlapping_text_info = []
        for t_seg in text_segments:
            t_start_sec = timedelta_to_seconds(t_seg['start_time'])
            t_end_sec = timedelta_to_seconds(t_seg['end_time'])
            
            if max(v_start_sec, t_start_sec) < min(v_end_sec, t_end_sec):
                overlapping_raw_scores.append(t_seg.get('final_text_score', 0))
                overlapping_text_info.append({
                    "text": t_seg.get('paragraph_text', ''),
                    "golden_quote": t_seg.get('analysis', {}).get('golden_quote_text', '')
                })

        # --- 动态归一化应用 ---
        if overlapping_raw_scores:
            avg_raw_text_score = sum(overlapping_raw_scores) / len(overlapping_raw_scores)
            if score_range == 0:
                normalized_text_score = 0.5 # 如果所有分数相同，给一个中间值
            else:
                normalized_text_score = (avg_raw_text_score - min_text_score) / score_range
        else:
            avg_raw_text_score = 0
            normalized_text_score = 0
            
        total_score = w_visual * v_score + w_text * normalized_text_score

        combined_results.append({
            "start": v_seg['start'],
            "end": v_seg['end'],
            "total_score": round(total_score, 4),
            "details": {
                "visual_score": v_score,
                "avg_text_score_raw": round(avg_raw_text_score, 4),
                "normalized_text_score": round(normalized_text_score, 4),
                "text_info": overlapping_text_info
            }
        })

    combined_results = sorted(combined_results, key=lambda x: x['total_score'], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    print(f"分数融合完成，已保存至 {output_path}")
    return output_path


def select_dynamic_highlights(all_scored_segments, min_duration,max_duration, std_dev_factor, max_cap):
    """根据一组动态规则来筛选高光片段。"""

    # 规则 1: 过滤掉时长过短的片段
    long_enough_segments = []
    for seg in all_scored_segments:
        start_sec = timedelta_to_seconds(seg['start'])
        end_sec = timedelta_to_seconds(seg['end'])
        if max_duration >= (end_sec - start_sec) >= min_duration:
            long_enough_segments.append(seg)

    print(f"  -> 规则1 (时长过滤): {len(all_scored_segments)} 个片段中，有 {len(long_enough_segments)} 个时长超过 {min_duration} 秒。")

    if not long_enough_segments:
        print("  -> 没有足够长的片段可选，流程终止。")
        return []

    # 规则 2: 只选择分数足够高的精英片段
    scores = [seg['total_score'] for seg in long_enough_segments]
    if len(scores) > 1:
        mean_score = statistics.mean(scores)
        stdev_score = statistics.stdev(scores)
#        score_threshold = mean_score - std_dev_factor * stdev_score
        score_threshold = 0.0
    else:
        # 如果只有一个片段，它就是唯一的选择
        score_threshold = scores[0] * 0.99

    elite_segments = [
        seg for seg in long_enough_segments if seg['total_score'] >= score_threshold
    ]

    print(f"  -> 规则2 (分数过滤): 平均分 {mean_score:.2f}，标准差 {stdev_score:.2f}，筛选门槛 {score_threshold:.2f}。")
    print(f"     有 {len(elite_segments)} 个精英片段脱颖而出。")

    # 规则 3: 应用数量上限（精英片段已经按分数排好序，直接取前 max_cap 个即可）
    final_selection = elite_segments[:max_cap]
    print(f"  -> 规则3 (数量上限): 最终选定 {len(final_selection)} 个高光片段进行导出。")

    return final_selection



def align_boundaries_to_semantics(highlight_clips, semantic_segments):
    """寻找与视觉高光重叠度最高的完整语义片段，并采用其边界。"""
    print("\n[边界对齐模块] 正在使用【最大重叠 V2】算法进行最终对齐...")

    aligned_clips = []

    if not semantic_segments:
        print("  -> 警告: 未找到任何语义片段，跳过边界对齐。")
        return highlight_clips

    for clip in highlight_clips:
        clip_start_sec = timedelta_to_seconds(clip['start'])
        clip_end_sec = timedelta_to_seconds(clip['end'])

        best_match_segment = None
        max_overlap = -1

        # 遍历所有句子，找到和当前高光片段重叠时间最长的那一句
        for sem_seg in semantic_segments:
            sem_start_sec = timedelta_to_seconds(sem_seg.get('start_time', '0:0:0'))
            sem_end_sec = timedelta_to_seconds(sem_seg.get('end_time', '0:0:0'))

            overlap_start = max(clip_start_sec, sem_start_sec)
            overlap_end = min(clip_end_sec, sem_end_sec)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_match_segment = sem_seg

        # 如果找到了最佳匹配的句子，就采用它的边界
        if best_match_segment:
            new_start_sec = timedelta_to_seconds(best_match_segment['start_time'])
            new_end_sec = timedelta_to_seconds(best_match_segment['end_time'])

            # 重新格式化时间字符串
            new_start_str = str(timedelta(seconds=new_start_sec)).split('.')[0] + '.' + f"{int((new_start_sec % 1) * 1000):03d}"
            new_end_str = str(timedelta(seconds=new_end_sec)).split('.')[0] + '.' + f"{int((new_end_sec % 1) * 1000):03d}"

            aligned_clip = clip.copy()
            aligned_clip['start'] = new_start_str
            aligned_clip['end'] = new_end_str
            aligned_clip['aligned_text'] = best_match_segment['paragraph_text']

            aligned_clips.append(aligned_clip)
            print(f"  -> 片段 [{clip['start']} -> {clip['end']}]")
            print(f"     最大重叠句子为: '{best_match_segment['paragraph_text'][:20]}...'")
            print(f"     最终对齐为 -> [{new_start_str} -> {new_end_str}]")
        else:
            aligned_clips.append(clip)
            print(f"  -> 片段 [{clip['start']} -> {clip['end']}] 未找到重叠句子，保留原始边界。")

    return aligned_clips





# === 主处理函数 ===
def execute_full_pipeline(task_id: str, video_url: str, api_keys: dict, configs: dict):
    """接收一个任务ID和视频URL，执行完整的视频分析和剪辑流程。"""

    DEEPSEEK_API_KEY = "sk-984f91a660ca40ab9427e513a97f67ca" 
    QWEN_API_KEY = "sk-0a0eefabc3f9421399d0f5981904326b"
    
    PYSCENE_SENSITIVITY = 50.0    

    MIN_CLIP_DURATION = 10.0
    MAX_CLIP_DURATION = 300.0
    SCORE_STD_DEV_FACTOR = 1.0
    MAX_CLIPS_CAP = 100


    
    # 1. 为本次任务创建一个独立的临时工作目录
    base_work_dir = "task_outputs"
    task_work_dir = os.path.join(base_work_dir, task_id)
    os.makedirs(task_work_dir, exist_ok=True)
    os.chmod(task_work_dir, 0o777)
    
    # 2. 下载视频
    original_video_path = download_video(video_url, task_work_dir)
    if not original_video_path:
        send_completion_callback(task_id, "failed", {"error": "Video download failed."})
        return

    # 3. 核心处理流程
    try:
        # --- 定义本次任务的所有中间文件路径 ---
        FRAMES_DIR = os.path.join(task_work_dir, "frames")
        MIXED_AUDIO_PATH = os.path.join(task_work_dir, "audio", "mixed_audio.wav")
        VOCALS_DIR = os.path.join(task_work_dir, "audio", "demucs_separated")
        TRANSCRIPT_PATH = os.path.join(task_work_dir, "segments", "transcript.json")
        SCENE_SEGMENTS_PATH = os.path.join(task_work_dir, "segments", "scene_segments.json")
        SEMANTIC_SEGMENTS_PATH = os.path.join(task_work_dir, "segments", "semantic_segments.json")
        VISUAL_SCORES_PATH = os.path.join(task_work_dir, "score", "visual_scores.json")
        TEXT_SCORES_PATH = os.path.join(task_work_dir, "score", "text_scores.json")
        COMBINED_SCORES_PATH = os.path.join(task_work_dir, "score", "combined_scores.json")
        HIGHLIGHTS_DIR = os.path.join(task_work_dir, "highlights")


        print(f"正在为任务 {task_id} 准备输出目录...")
        for path in [FRAMES_DIR, os.path.dirname(MIXED_AUDIO_PATH), VOCALS_DIR, os.path.dirname(TRANSCRIPT_PATH), os.path.dirname(VISUAL_SCORES_PATH), HIGHLIGHTS_DIR]:
            os.makedirs(path, exist_ok=True)
            os.chmod(task_work_dir, 0o777)
        print("输出目录准备完毕。")


        # ===================================================================
        #                        【工作流开始】
        # ===================================================================
        print("==========阶段0:标准化视频格式 =========")
        video_input_path = convert_to_mp4(original_video_path)
        print(f"视频已标准化为:{video_input_path}")

       
        print("========== 阶段 1: 预处理与人声分离 ==========")
        extract_frames(video_input_path, FRAMES_DIR)
        extract_audio(video_input_path, MIXED_AUDIO_PATH)
        clean_vocals_path = separate_vocals(MIXED_AUDIO_PATH, VOCALS_DIR)
    
        if not clean_vocals_path:
            print("严重错误：人声分离失败，将使用原始混合音轨进行后续步骤。")
            clean_vocals_path = MIXED_AUDIO_PATH

        print("\n========== 阶段 2: 视觉与文本分析 (并行) ==========")
        print("--- 视觉分析流 ---")
        detect_scene_changes(video_input_path, SCENE_SEGMENTS_PATH, threshold=PYSCENE_SENSITIVITY)
        run_visual_scoring_pipeline(api_key=QWEN_API_KEY, frame_dir=FRAMES_DIR, segment_file=SCENE_SEGMENTS_PATH, output_file=VISUAL_SCORES_PATH)
    

        print("\n--- 文本分析流 ---")
        transcribe_audio(clean_vocals_path, TRANSCRIPT_PATH)
        semantic_segment_final(TRANSCRIPT_PATH, SEMANTIC_SEGMENTS_PATH, api_key=DEEPSEEK_API_KEY)
        run_scoring_pipeline(SEMANTIC_SEGMENTS_PATH, TEXT_SCORES_PATH, api_key=DEEPSEEK_API_KEY)


        print("\n========== 阶段 3: 核心分数融合 ==========")
        all_segments_path = fuse_scores(VISUAL_SCORES_PATH, TEXT_SCORES_PATH, COMBINED_SCORES_PATH)
        with open(all_segments_path, 'r', encoding='utf-8') as f:
            all_scored_segments = json.load(f)

        print("\n========== 阶段 4: 动态筛选高光片段 ==========")
        candidate_highlight_segments = select_dynamic_highlights(
            all_scored_segments=all_scored_segments,
            min_duration=MIN_CLIP_DURATION,
            std_dev_factor=SCORE_STD_DEV_FACTOR,
            max_cap=MAX_CLIPS_CAP,
            max_duration=MAX_CLIP_DURATION
        )

        if not candidate_highlight_segments:
            print("未能根据筛选规则找到任何合适的高光片段。")
            print("\n========== 所有任务已完成！ ==========")
            return
    

        print("\n========== 阶段 5: 语义边界对齐 ==========")
        try:
            with open(SEMANTIC_SEGMENTS_PATH, 'r', encoding='utf-8') as f:
                semantic_segments = json.load(f)

            aligned_clips = align_boundaries_to_semantics(
                candidate_highlight_segments, 
                semantic_segments
            )
        except FileNotFoundError:
            print(f"警告：找不到语义分段文件 {SEMANTIC_SEGMENTS_PATH}，将跳过边界对齐步骤。")
            aligned_clips = candidate_highlight_segments


        print("\n========== 阶段 5.5: 最终质检与去重 ==========")
        final_unique_clips = []
        seen_boundaries = set()

        for clip in aligned_clips:
            # 检查1: 时长是否满足最低要求
            start_sec = timedelta_to_seconds(clip['start'])
            end_sec = timedelta_to_seconds(clip['end'])
            if (end_sec - start_sec) < MIN_CLIP_DURATION:
                print(f"  -> 质检淘汰: 片段 [{clip['start']} -> {clip['end']}] (时长 {(end_sec - start_sec):.1f}s) 短于设定的最短时长 {MIN_CLIP_DURATION}s。")
                continue # 跳过这个不合格的片段

            #规则1.5: 时长是否超过最大允许
            if (end_sec - start_sec) > MAX_CLIP_DURATION:
                print(f"  -> 质检淘汰: 片段 [{clip['start']} -> {clip['end']}] (时长 {(end_sec - start_sec):.1f}s) 超过最大时长 {MAX_CLIP_DURATION}s。")
                continue  # 跳过这个超长片段
            

            # 检查2: 是否与已选中的片段重复
            boundary_key = (clip['start'], clip['end'])
            if boundary_key in seen_boundaries:
                print(f"  -> 质检淘汰: 片段 [{clip['start']} -> {clip['end']}] 是一个重复片段。")
                continue # 跳过这个重复的片段

            final_unique_clips.append(clip)
            seen_boundaries.add(boundary_key)

        print(f"  -> 经过最终质检后，剩余 {len(final_unique_clips)} 个合格的唯一片段。")


        print("\n========== 阶段 6: 导出高光片段与智能解释 ==========")
        export_final_clips(
            segments_to_export=final_unique_clips, 
            original_video_path=video_input_path,
            frames_dir=FRAMES_DIR,
            output_dir=HIGHLIGHTS_DIR,
            qwen_api_key=QWEN_API_KEY
        )
       
        # ==========================================================
        print("\n========== 阶段 7: 开始为每个高光片段生成 9:16 竖屏版本 ==========")
        try:
            # 1. 找出所有刚刚在阶段6中生成的高光片段文件
            original_highlight_files = [f for f in os.listdir(HIGHLIGHTS_DIR) if f.endswith('.mp4') and '_9x16_vertical' not in f]
            print(f"找到 {len(original_highlight_files)} 个原始高光片段准备进行转换。")

            # 2. 遍历每个高光片段，并为其创建对应的9:16版本
            for clip_filename in original_highlight_files:
                original_clip_path = os.path.join(HIGHLIGHTS_DIR, clip_filename)
                
                # 构造新生成的竖屏视频的文件名
                file_name, file_ext = os.path.splitext(clip_filename)
                vertical_clip_output_path = os.path.join(HIGHLIGHTS_DIR, f"{file_name}_9x16_vertical{file_ext}")
                
                print(f"\n--- 正在转换: {clip_filename} ---")
                
                # 调用我们封装好的智能转换函数
                reframe_to_vertical_video(
                    input_video_path=original_clip_path,
                    output_video_path=vertical_clip_output_path
                )
                print(f"--- 完成转换: {clip_filename} ---")

        except Exception as reframe_error:
            print(f"错误：在阶段 7 (智能竖屏转换) 中发生错误: {reframe_error}")


        print("\n========== 所有任务已完成！ ==========")
        print(f" 任务 {task_id} 处理成功。")
       

        # ==========================================================
        # --- 任务成功，准备包含【内网URL】的回调结果 ---
        SERVER_PUBLIC_IP = "127.0.0.1"
        BASE_URL = f"http://{SERVER_PUBLIC_IP}"

        try:
            # 找出所有原始高光片段（不包含我们新生成的竖屏版）
            original_files = [f for f in os.listdir(HIGHLIGHTS_DIR) if f.endswith('.mp4') and '_9x16_vertical' not in f]
            
            # 创建一个列表，用来存放每一组视频（原始+竖屏）的URL
            clips_data = []

            for original_file in original_files:
                # 原始视频的URL
                original_url = f"{BASE_URL}/task_outputs/{task_id}/highlights/{original_file}"
                
                # 查找其对应的竖屏版本是否存在
                file_name, file_ext = os.path.splitext(original_file)
                vertical_filename = f"{file_name}_9x16_vertical{file_ext}"
                vertical_filepath = os.path.join(HIGHLIGHTS_DIR, vertical_filename)

                vertical_url = None # 默认为空
                if os.path.exists(vertical_filepath):
                    vertical_url = f"{BASE_URL}/task_outputs/{task_id}/highlights/{vertical_filename}"
                
                # 将这一组URL作为一个对象添加到列表中
                clips_data.append({
                    "original_clip_url": original_url,
                    "vertical_9x16_clip_url": vertical_url
                })

            final_results = {
                "message": "Processing completed successfully.",
                "clips": clips_data, # 使用新的、结构化的列表
                "browse_results_url": f"{BASE_URL}/task_outputs/{task_id}/"
            }
        except Exception as result_error:
            final_results = {"message": "Processing completed, but failed to generate result URLs.", "error": str(result_error)}

        send_completion_callback(task_id, "completed", final_results)


    except Exception as e:
        # --- 5. 任何步骤出错，捕获异常并发送失败回调 ---
        print(f"错误：任务 {task_id} 处理失败。错误: {e}")
        send_completion_callback(task_id, "failed", {"error": str(e)})
    
