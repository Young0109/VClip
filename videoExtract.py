import os
import subprocess
import sys

def extract_frames(video_path, output_dir="frames", fps=1):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f" 视频文件不存在：{video_path}")

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
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
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",                # 标准无损音频采集格式 （PCM）
        "-ar", "44100",                        # 采样率 44100khz
        "-ac", "2",                            # 双声道（立体声）
        output_audio_path
    ]
    subprocess.run(cmd, check=True)
    print(f" 音频提取完成，输出路径：{output_audio_path}")

# === 这里是刚才缺失的函数 ===
def convert_to_mp4(input_path):
    """将视频转换为标准 MP4 格式，确保兼容性"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    # 定义输出文件名
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name_no_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(dir_name, f"{name_no_ext}_standard.mp4")
    
    print(f"正在标准化视频格式: {input_path} -> {output_path}")

    # 使用 ffmpeg 进行转码
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",  # 强制使用 H.264 编码
        "-c:a", "aac",      # 强制使用 AAC 音频
        "-strict", "experimental",
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    print(f"视频格式标准化完成: {output_path}")
    return output_path
# ===========================

def preprocess_video(video_path, frame_dir="frames", audio_path="audio/audio.wav"):
    print(f"开始预处理：{video_path}")
    # 可以在这里加一步标准化，或者在 core_logic 里调用
    extract_frames(video_path, frame_dir)
    extract_audio(video_path, audio_path)
    print("视频预处理完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python videoExtract.py your_video.mp4")
    else:
        preprocess_video(sys.argv[1])
