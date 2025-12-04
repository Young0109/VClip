import subprocess
import os
import torch
import sys

def separate_vocals(input_audio_path: str, output_dir: str) -> str:
    """使用 Demucs 将音轨分离为人声和背景音。"""
    
    if not os.path.exists(input_audio_path):
        print(f"错误：找不到输入的音频文件 '{input_audio_path}'")
        return None

    print(f"\n[人声分离模块] 正在使用 Demucs 分离音频: {os.path.basename(input_audio_path)}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  -> Demucs 将使用设备: {device}")

    """
    cmd = [
        "python", "-m", "demucs.separate",
        "--two-stems=vocals",
        "-d", device,
        "-o", output_dir,
        input_audio_path
    ]
    """

    cmd = [
        sys.executable, "-m", "demucs.separate",  # ✅ 用当前解释器
        "--two-stems=vocals",
        "-d", device,
        "-o", output_dir,
        input_audio_path
    ]

    
    try:
        print(f"  -> 正在执行 Demucs 命令...")
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print("  -> Demucs 命令执行完毕。")

        # --- 在这里寻找正确的输出文件路径 ---
        model_output_dir_name = next((d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))), None)
        if not model_output_dir_name:
            print(f"错误：在 {output_dir} 中找不到 Demucs 创建的模型输出目录。")
            return None
        
        model_output_dir_path = os.path.join(output_dir, model_output_dir_name)
        original_filename_base = os.path.splitext(os.path.basename(input_audio_path))[0]
        vocals_path = os.path.join(model_output_dir_path, original_filename_base, "vocals.wav")
        
        if os.path.exists(vocals_path):
            print(f"  -> 人声分离成功！干净的人声文件已保存至: {vocals_path}")
            return vocals_path
        else:
            print(f"错误：人声分离看似已完成，但在预期路径找不到输出的 vocals.wav 文件。")
            print(f"预期路径是: {vocals_path}")
            return None

    except subprocess.CalledProcessError as e:
        print("错误：Demucs 执行失败。")
        print("Demucs Stderr:", e.stderr)
        return None

    except Exception as e:
        print(f"未知错误发生: {e}")
        return None
