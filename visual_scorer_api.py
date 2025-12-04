import os
import json
import re
from datetime import timedelta
import dashscope
from dashscope import MultiModalConversation

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
            print(f"API 响应异常: {response}")
            return "emotion=0.0, impact=0.0, info=0.0"

    except Exception as e:
        print(f"API 调用异常: {e}")
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
    print(f"警告: 无法从文本中解析分数: '{text}'")
    return {}

def timedelta_str_to_seconds(time_str: str) -> int:
    """将 'HH:MM:SS' 格式的时间字符串转换为整数秒。"""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s_part = parts
        s = s_part.split('.')[0] 
        return int(h) * 3600 + int(m) * 60 + int(s)
    return 0

def run_visual_scoring_pipeline(api_key: str, frame_dir: str, segment_file: str, output_file: str):
    """视觉评分主流程函数。它接收外部传入的参数，对分段后的视频帧进行AI评分。"""
    # 1. 使用传入的参数来配置API Key
    dashscope.api_key = api_key
    
    PROMPT = (
        "请判断以下画面中人物的情绪强度（0-1）、画面吸引力（0-1）和信息密度（0-1），"
        "并说明是否有字幕、关键人物或图表。严格按照此格式输出，不要有任何其他文字：emotion=数值, impact=数值, info=数值"
    )

    # 2. 读取传入路径的 segment 文件
    if not os.path.exists(segment_file):
        print(f"错误: 找不到场景分段文件 '{segment_file}'")
        return
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(segment_file, "r", encoding='utf-8') as f:
        segments = json.load(f)

    # 3. 从传入的 frame_dir 中获取帧文件列表
    if not os.path.exists(frame_dir):
        print(f"错误: 找不到帧目录 '{frame_dir}'")
        return
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    if not frame_files:
        print(f"错误: 帧目录 '{frame_dir}' 为空，没有任何.jpg图片。")
        return
        
    results = []

    print(f"开始对 {len(segments)} 个视觉片段进行评分...")
    for i, segment in enumerate(segments):
        start_sec = timedelta_str_to_seconds(segment["start"])
        end_sec = timedelta_str_to_seconds(segment["end"])
        
        # 使用中间帧作为代表
        middle_sec = (start_sec + end_sec) // 2
        
        # 帧的文件名是从 1 开始编号的，所以秒数直接对应帧号
        frame_index_to_find = middle_sec 
        frame_filename_to_find = f"{frame_index_to_find:06d}.jpg"
        
        frame_path = os.path.join(frame_dir, frame_filename_to_find)

        # 如果找不到精确的中间帧，就用最接近的起始帧
        if not os.path.exists(frame_path):
            frame_index_to_find = start_sec 
            frame_filename_to_find = f"{frame_index_to_find:06d}.jpg"
            frame_path = os.path.join(frame_dir, frame_filename_to_find)
            if not os.path.exists(frame_path):
                print(f"  警告: 找不到片段 {segment['start']} -> {segment['end']} 的任何代表帧，跳过此片段。")
                continue

        print(f"  [{i+1}/{len(segments)}] 正在分析片段 {segment['start']} -> {segment['end']} (使用帧: {frame_filename_to_find})")

        # 调用API
        response_text = call_qwen_vl(frame_path, PROMPT)
        scores = extract_scores_from_text(response_text)
        
        # 计算总分并整合结果
        scores.update({
            "start": segment["start"],
            "end": segment["end"],
            "total_visual_score": round(
                0.4 * scores.get("emotion_score", 0) +
                0.4 * scores.get("impact_score", 0) +
                0.2 * scores.get("info_density", 0), 3)
        })
        results.append(scores)

    # 4. 将结果保存到传入的 output_file 路径
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"视觉评分完成，结果已保存至 {output_file}")



