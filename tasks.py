
from celery import Celery
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from service.core_logic import execute_full_pipeline
from service.database import update_task_status, update_task_as_failed

# --- Celery 应用配置 ---
REDIS_URL = "redis://localhost:6379/0"
celery_app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True
)

# --- 异步任务 ---

@celery_app.task(name="tasks.process_video_task", bind=True)
def process_video_task(self, task_id: str, video_url: str, api_keys: dict, configs: dict):
    """ Celery 异步任务，负责启动处理并更新初始和最终异常状态。 """

    try:
        # 步骤1: 任务开始，立刻更新数据库状态为“处理中”
        update_task_status(task_id, 'processing', '任务已由后台Worker接收，正在处理中...')
        
        # 步骤2: 调用核心处理引擎
        execute_full_pipeline(
            task_id=task_id,
            video_url=video_url,
            api_keys=api_keys,
            configs=configs
        )
        
        # 如果 execute_full_pipeline 顺利执行完毕（没有抛出异常）, Celery 任务本身就算成功完成了。 最终的 "completed" 状态将由 FastAPI 回调来更新。
        return {"status": "dispatched", "task_id": task_id}
        
    except Exception as e:
        # 步骤3: 充当保险丝. 如果 execute_full_pipeline 本身发生严重崩溃，没来得及发送失败回调， 我们在这里捕获这个异常，并强制更新数据库状态为“失败”。
        error_message = f"Celery task encountered a critical failure: {e}"
        print(error_message)
        update_task_as_failed(task_id, error_message)
        
        raise e
