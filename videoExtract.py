import os
import subprocess
import sys
import shutil

# ====================================================
# 👇👇👇 【已预填】你的 FFmpeg 路径 👇👇👇
FFMPEG_BIN = r"E:\VClip依赖\ffmpeg-2025-12-04-git-d6458f6a8b-full_build\bin\ffmpeg.exe"
# ====================================================

# 自动检查路径
if not os.path.exists(FFMPEG_BIN):
    print(f"⚠️ 警告: 路径 {FFMPEG_BIN} 不存在！")
    if shutil.which("ffmpeg"):
        FFMPEG_BIN = "ffmpeg"
    else:
        print("❌ 严重错误: 系统找不到 ffmpeg！")

print(f"✅ 当前使用的 FFmpeg 路径: {FFMPEG_BIN}")

def extract_frames(video_path, output_dir="frames", fps=1):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f" 视频文件不存在：{video_path}")

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-i", video_path,
        "-vf", f"fps={fps}",
        f"{output_dir}/%06d.jpg"
    ]
    subprocess.run(cmd, check=True)
    print(f" 帧提取完成，保存目录：{output_dir}")

def extract_audio(video_path, output_audio_path="audio/audio.wav"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f" 视频文件不存在：{video_path}")

    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_audio_path
    ]
    subprocess.run(cmd, check=True)
    print(f" 音频提取完成，输出路径：{output_audio_path}")

def convert_to_mp4(input_path):
    """将视频转换为标准 MP4 格式 (使用 RTX 5080 NVENC 加速)"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name_no_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(dir_name, f"{name_no_ext}_standard.mp4")
    
    print(f"正在标准化视频格式 (GPU加速): {input_path} -> {output_path}")

    # 使用 ffmpeg 进行转码
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        
        # --- 👇 核心修改在这里 👇 ---
        "-c:v", "h264_nvenc",   # 使用 NVIDIA 显卡硬件编码
        "-preset", "p4",        # 预设：p1(最快) ~ p7(最慢/质量最好)，p4 是平衡点
        "-b:v", "5M",           # 设置视频码率为 5Mbps，保证清晰度
        # -------------------------
        
        "-c:a", "aac",      # 音频使用 AAC
        "-strict", "experimental",
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    print(f"视频格式标准化完成: {output_path}")
    return output_path

def preprocess_video(video_path, frame_dir="frames", audio_path="audio/audio.wav"):
    print(f"开始预处理：{video_path}")
    extract_frames(video_path, frame_dir)
    extract_audio(video_path, audio_path)
    print("视频预处理完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python videoExtract.py your_video.mp4")
    else:
        preprocess_video(sys.argv[1])