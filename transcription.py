import os
from datetime import timedelta
import whisper
import torch
import json

def format_seconds_to_hms(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

def transcribe_audio(audio_path: str, output_json_path: str):
    """接收一个音频文件路径，使用 Whisper 进行转写，并保存结果。"""
    
    if not os.path.exists(audio_path):
        print(f"错误：找不到要转写的音频文件 '{audio_path}'")
        return

    print(f"\n[语音转写模块] 开始转写音频文件: {os.path.basename(audio_path)}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  -> Whisper 将使用设备: {device}")

    model = whisper.load_model("medium", device=device)
    print("  -> Whisper 'medium' 模型加载完毕。")

    SIMPLIFIED_CHINESE_PROMPT = "以下是普通话的句子，请用简体字转写。"
    
    print("  -> 正在进行语音识别...")
    result = model.transcribe(
        audio_path, 
        language='Chinese', 
        word_timestamps=True,
        initial_prompt=SIMPLIFIED_CHINESE_PROMPT
    )
    print("  -> 音频转写完成。")

    formatted_segments = []
    for segment in result['segments']:
        formatted_segments.append({
            "start_time": format_seconds_to_hms(segment['start']),  
            "end_time": format_seconds_to_hms(segment['end']),  
            "text": segment['text'].strip()
        })
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_segments, f, ensure_ascii=False, indent=4)

    print(f"  -> 转写文稿已保存至: {output_json_path}")
