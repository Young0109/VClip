# 文件路径: E:\VClip V1.1\tasks.py (确保是根目录这个)

from celery import Celery
import os
import sys

# 获取当前文件所在的目录（即根目录）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 将根目录加入 Python 搜索路径
sys.path.insert(0, CURRENT_DIR)

# 【这里是关键】
# 以前是: from service.core_logic import ...
# 现在改为:
from core_logic import execute_full_pipeline
from database import update_task_status, update_task_as_failed

# --- Celery 应用配置 ---
# 确保 Redis 地址是对的，如果是在 Docker 或 WSL 里可能不一样
REDIS_URL = "redis://localhost:6379/0"

celery_app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True
)

@celery_app.task(name="tasks.process_video_task", bind=True)
def process_video_task(self, task_id: str, video_url: str, api_keys: dict, configs: dict):
    try:
        # 这里也需要去 core_logic.py 确认 update_task_status 导入是否正确
        update_task_status(task_id, 'processing', '任务处理中...')
        
        execute_full_pipeline(
            task_id=task_id,
            video_url=video_url,
            api_keys=api_keys,
            configs=configs
        )
        return {"status": "dispatched", "task_id": task_id}
        
    except Exception as e:
        error_msg = f"Task Failed: {e}"
        print(error_msg)
        update_task_as_failed(task_id, error_msg)
        raise e