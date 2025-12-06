import subprocess
import os
import torch
import sys
import shutil

# ====================================================
# 👇👇👇 【已预填】你的 FFmpeg 路径 👇👇👇
FFMPEG_PATH_INPUT = r"E:\VClip依赖\ffmpeg-2025-12-04-git-d6458f6a8b-full_build\bin\ffmpeg.exe"
# ====================================================

# --- 1. 智能 FFmpeg 路径修正与环境变量注入 ---
if FFMPEG_PATH_INPUT.lower().endswith(".exe"):
    FFMPEG_DIR = os.path.dirname(FFMPEG_PATH_INPUT)
else:
    FFMPEG_DIR = FFMPEG_PATH_INPUT

if os.path.exists(FFMPEG_DIR):
    if FFMPEG_DIR not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + FFMPEG_DIR
        print(f"✅ [人声分离] 已将 FFmpeg 目录加入环境变量: {FFMPEG_DIR}")
else:
    print(f"❌ [人声分离] 严重警告: 填写的路径不存在: {FFMPEG_DIR}")


def separate_vocals(input_audio_path: str, output_dir: str) -> str:
    """使用 Demucs 将音轨分离为人声和背景音。"""
    
    if not os.path.exists(input_audio_path):
        print(f"错误：找不到输入的音频文件 '{input_audio_path}'")
        return None

    print(f"\n[人声分离模块] 正在使用 Demucs 分离音频: {os.path.basename(input_audio_path)}...")
    
    # --- 2. 恢复自动检测 CUDA (RTX 5080 火力全开模式) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  -> Demucs 将使用设备: {device} (显卡加速中 🚀)")

    # 构造命令
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "--two-stems=vocals",
        "-d", device,
        "-o", output_dir,
        input_audio_path
    ]
    
    try:
        print(f"  -> 正在执行 Demucs 命令...")
        
        # 【关键修改】保留了 errors='replace'，防止乱码导致崩溃
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            errors='replace'  # <--- 遇到乱码字符自动忽略
        )
        print("  -> Demucs 命令执行完毕。")

        # --- 3. 寻找输出文件 ---
        model_name = "htdemucs" 
        song_name = os.path.splitext(os.path.basename(input_audio_path))[0]
        vocals_path = os.path.join(output_dir, model_name, song_name, "vocals.wav")
        
        if os.path.exists(vocals_path):
            print(f"  -> 人声分离成功！文件保存至: {vocals_path}")
            return vocals_path
        
        print(f"  -> 标准路径未找到，正在搜索 output 目录...")
        for root, dirs, files in os.walk(output_dir):
            if "vocals.wav" in files:
                found_path = os.path.join(root, "vocals.wav")
                print(f"  -> 在子目录找到了: {found_path}")
                return found_path
                
        print(f"错误：Demucs 执行成功，但在 {output_dir} 及其子目录中没找到 vocals.wav。")
        return None

    except subprocess.CalledProcessError as e:
        print("错误：Demucs 执行失败。")
        print("====== Demucs 报错详情 ======")
        print(e.stderr) 
        print("============================")
        return None

    except Exception as e:
        print(f"未知错误发生: {e}")
        return None