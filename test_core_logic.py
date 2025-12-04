import sys
import os
import uuid 

sys.path.append(os.path.abspath('service'))
from core_logic import execute_full_pipeline

def run_test():

    print("--- 开始独立测试 core_logic 模块 ---")

    # --- 1. 准备测试数据 ---
    test_task_id = f"test_{uuid.uuid4().hex[:8]}"

    test_video_url = "https://cats.yunshicloud.com/resource_tts_153/materialds/20250315/f3b54cd1-5c50-45c2-9e69-9aa8adefecf4_transcoded.mp4" 

    test_api_keys = {
        "deepseek": "sk-984f91a660ca40ab9427e513a97f67ca" ,
        "qwen": "sk-0a0eefabc3f9421399d0f5981904326b"
    }
    
    test_configs = {
        "histogram_threshold": 0.2,
        "min_clip_duration": 10.0, 
        "score_std_dev_factor": 0.4,
        "max_clips_cap": 30
    }

    print(f"测试任务ID: {test_task_id}")
    print(f"测试视频URL: {test_video_url}")

    # --- 2. 调用核心处理函数 ---
    execute_full_pipeline(
        task_id=test_task_id,
        video_url=test_video_url,
        api_keys=test_api_keys,
        configs=test_configs
    )

    print("\n--- 核心逻辑模块独立测试结束 ---")


if __name__ == "__main__":
    run_test()
