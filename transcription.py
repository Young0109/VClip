import os
from datetime import timedelta
import whisper
import torch
import json

# ====================================================
# 👇👇👇 【必须修改】填入你 FFmpeg 的 bin 文件夹路径 👇👇👇
# 这里的路径要和 vocal_separator.py 里填的一样
# ====================================================
FFMPEG_DIR = r"E:\VClip依赖\ffmpeg-2025-12-04-git-d6458f6a8b-full_build\bin"
# ====================================================

# 【关键修复】把 FFmpeg 临时加入系统环境变量，让 Whisper 能找到它
if os.path.exists(FFMPEG_DIR):
    os.environ["PATH"] += os.pathsep + FFMPEG_DIR
else:
    print(f"⚠️ 警告: FFmpeg 路径不存在: {FFMPEG_DIR}，Whisper 可能会报错 [WinError 2]")

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

    # --- 强制使用 CPU (适配 RTX 5080 兼容性问题) ---
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"  -> Whisper 将使用设备: {device}")

    try:
        model = whisper.load_model("medium", device=device)
        print("  -> Whisper 'medium' 模型加载完毕。")

        SIMPLIFIED_CHINESE_PROMPT = "以下是普通话的句子，请用简体字转写。"
        
        print("  -> 正在进行语音识别...")
        
        # Whisper 在这里会调用 ffmpeg，如果环境变量没设对，就会报 [WinError 2]
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

    except Exception as e:
        print(f"❌ Whisper 转写过程中发生错误: {e}")
        # 这里不抛出异常，为了防止整个 Worker 崩溃，但实际生产中应该抛出
        raise e