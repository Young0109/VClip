import os
import json
import re
from datetime import timedelta
import dashscope
from dashscope import MultiModalConversation
import concurrent.futures
import time

# --- 1. 基础辅助函数保持不变 ---

def call_qwen_vl(image_path, prompt):
    """调用通义千问API进行图片分析。"""
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        print(f"Image not found: {abs_path}")
        return "emotion=0.0, impact=0.0, info=0.0"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": f"file://{abs_path}"} 
        ]
    }]

    try:
        # 增加简单的重试机制，防止网络瞬间抖动
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = MultiModalConversation.call(
                    model="qwen-vl-plus",
                    messages=messages
                )
                if response and response.status_code == 200 and "output" in response and "choices" in response["output"]:
                    content_list = response["output"]["choices"][0]["message"]["content"]
                    if isinstance(content_list, list):
                        content_text = "".join([part.get("text", "") for part in content_list])
                    else:
                        content_text = str(content_list)
                    return content_text
                else:
                    print(f"API 响应异常 (尝试 {attempt+1}/{max_retries}): {response}")
            except Exception as e:
                print(f"API 调用连接错误 (尝试 {attempt+1}/{max_retries}): {e}")
            
            time.sleep(1) # 失败后稍等一秒

        return "emotion=0.0, impact=0.0, info=0.0"

    except Exception as e:
        print(f"API 调用最终失败: {e}")
        return "emotion=0.0, impact=0.0, info=0.0"

def extract_scores_from_text(text: str) -> dict:
    """从API返回的文本中提取分数。"""
    pattern = r"emotion=([0-9.]+).*impact=([0-9.]+).*info=([0-9.]+)"
    match = re.search(pattern, text.replace(" ", "").replace("\n", ""))
    if match:
        return {
            "emotion_score": float(match.group(1)),
            "impact_score": float(match.group(2)),
            "info_density": float(match.group(3)),
        }
    # 如果提取失败，尝试宽松匹配或给默认值
    return {
        "emotion_score": 0.0,
        "impact_score": 0.0,
        "info_density": 0.0
    }

def timedelta_str_to_seconds(time_str: str) -> int:
    """将 'HH:MM:SS' 格式的时间字符串转换为整数秒。"""
    if not time_str: return 0
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s_part = parts
        s = s_part.split('.')[0] 
        return int(h) * 3600 + int(m) * 60 + int(s)
    return 0

# --- 2. 新增：单个片段的处理逻辑 ---

def process_single_segment(segment, index, frame_dir, prompt):
    """处理单个视频片段的线程函数"""
    try:
        start_sec = timedelta_str_to_seconds(segment["start"])
        end_sec = timedelta_str_to_seconds(segment["end"])
        
        # 使用中间帧作为代表
        middle_sec = (start_sec + end_sec) // 2
        
        # 帧的文件名是从 1 开始编号的（假设 videoExtract 生成的是 000001.jpg 对应第1秒）
        # 这里需要根据你的 videoExtract 逻辑确认，通常 fps=1 时，第N秒对应第N张或N+1张
        # 假设文件名是 6位数字.jpg
        frame_filename_to_find = f"{middle_sec:06d}.jpg"
        if middle_sec == 0: frame_filename_to_find = "000001.jpg" # 修正第0秒的情况

        frame_path = os.path.join(frame_dir, frame_filename_to_find)

        # 如果找不到精确的中间帧，就用最接近的起始帧
        if not os.path.exists(frame_path):
            alt_index = max(1, start_sec)
            frame_filename_to_find = f"{alt_index:06d}.jpg"
            frame_path = os.path.join(frame_dir, frame_filename_to_find)
            
            if not os.path.exists(frame_path):
                print(f"  [线程-{index}] 警告: 找不到片段 {segment['start']} 的任何代表帧，跳过。")
                return None

        # print(f"  [线程-{index}] 正在分析: {frame_filename_to_find} ...")

        # 调用API
        response_text = call_qwen_vl(frame_path, prompt)
        scores = extract_scores_from_text(response_text)
        
        # 计算总分并整合结果
        scores.update({
            "start": segment["start"],
            "end": segment["end"],
            "total_visual_score": round(
                0.4 * scores.get("emotion_score", 0) +
                0.4 * scores.get("impact_score", 0) +
                0.2 * scores.get("info_density", 0), 3),
            "original_index": index # 记录原始索引，方便后续排序
        })
        
        print(f"  [线程-{index}] 完成: visual_score={scores['total_visual_score']}")
        return scores

    except Exception as e:
        print(f"  [线程-{index}] 处理异常: {e}")
        return None

# --- 3. 主流程：改为多线程并行 ---

def run_visual_scoring_pipeline(api_key: str, frame_dir: str, segment_file: str, output_file: str):
    """视觉评分主流程函数（多线程并行版）。"""
    
    dashscope.api_key = api_key
    
    PROMPT = (
        "请判断以下画面中人物的情绪强度（0-1）、画面吸引力（0-1）和信息密度（0-1），"
        "并说明是否有字幕、关键人物或图表。严格按照此格式输出，不要有任何其他文字：emotion=数值, impact=数值, info=数值"
    )

    if not os.path.exists(segment_file):
        print(f"错误: 找不到场景分段文件 '{segment_file}'")
        return
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(segment_file, "r", encoding='utf-8') as f:
        segments = json.load(f)

    if not os.path.exists(frame_dir):
        print(f"错误: 找不到帧目录 '{frame_dir}'")
        return

    # === 配置并发数量 ===
    # 建议设置在 5-10 之间。太高可能会触发 API 的 QPS 限制（每秒请求限制）导致报错。
    # 如果你是付费的高级版 API，可以尝试设为 10 或更高。
    MAX_WORKERS = 8 
    
    print(f"开始对 {len(segments)} 个视觉片段进行评分 (并发数: {MAX_WORKERS})...")
    start_time = time.time()

    results = []
    
    # 使用 ThreadPoolExecutor 管理线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_single_segment, seg, i, frame_dir, PROMPT): i 
            for i, seg in enumerate(segments)
        }
        
        # 获取结果 (as_completed 会在某个任务一完成就立刻返回，顺序是乱的)
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as exc:
                print(f"  任务 {idx} 抛出异常: {exc}")

    # === 关键：恢复顺序 ===
    # 因为多线程完成的顺序是不确定的，我们需要按原始索引重新排序
    results.sort(key=lambda x: x['original_index'])
    
    # 移除临时的索引字段，保持输出干净
    for r in results:
        r.pop('original_index', None)

    elapsed_time = time.time() - start_time
    print(f"视觉评分完成，耗时 {elapsed_time:.2f} 秒。结果已保存至 {output_file}")

    # 4. 保存结果
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
