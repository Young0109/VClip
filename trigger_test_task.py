
import sys
import os
import uuid

sys.path.append(os.path.abspath('service'))

from database import create_task
from tasks import process_video_task

def trigger():
    """创建一个任务，并将其发送到 Celery 队列。"""

    test_video_url = "https://cats.yunshicloud.com/resource_tts_153/materialds/20250315/f3b54cd1-5c50-45c2-9e69-9aa8adefecf4_transcoded.mp4" 

    test_api_keys = {
        "deepseek": "sk-984f91a660ca40ab9427e513a97f67ca" ,
        "qwen": "sk-0a0eefabc3f9421399d0f5981904326b"
    }

    test_configs = {
        "histogram_threshold": 0.5,
        "min_clip_duration": 10.0, 
        "score_std_dev_factor": 0.5,
        "max_clips_cap": 30
    }

    # 1. 首先，在数据库中创建任务，获取 task_id
    task_id = create_task(test_video_url)

    if task_id:
        print(f"成功在数据库中创建任务，ID为: {task_id}")
        print(f"将使用视频 URL: {test_video_url}")

        # 2. 然后，将这个任务和所有参数发送到 Celery 队列
        process_video_task.delay(
            task_id=task_id,
            video_url=test_video_url,
            api_keys=test_api_keys,
            configs=test_configs
        )
        print(f"任务 {task_id} 已成功发送至 Celery 队列。")
        print("\n请立即切换到另一个终端查看 Worker 的处理日志！")
    else:
        print("在数据库中创建任务失败，测试终止。")

if __name__ == "__main__":
    trigger()
