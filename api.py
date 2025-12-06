
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict, Any 
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sys
import os
import shutil
import uuid

# 【修改点】删除了 sys.path.append，直接导入
from database import create_task as db_create_task
from database import get_task as db_get_task
from database import update_task_as_completed, update_task_as_failed
from tasks import process_video_task

app = FastAPI(root_path="/vc")


# ==========================================================


# --- API 数据模型 (Pydantic) ---

# 【修正】请求模型，允许 configs 接收任意类型的值
class TaskRequest(BaseModel):
    video_url: str
    api_keys: Optional[Dict[str, str]] = None
    configs: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str

# 【修正】回调负载模型，使其与 core_logic.py 发送的数据结构完全匹配
class CallbackPayload(BaseModel):
    task_id: str
    status: str
    results: Dict[str, Any]


# --- API 接口定义 ---

@app.post("/tasks/", response_model=TaskResponse, status_code=202)
def submit_task(req: TaskRequest):
    """接收新的视频处理任务，并将其推送到Celery队列。"""
    
    # 【优化】调用 database.py 中的函数来创建任务
    task_id = db_create_task(req.video_url)
    if not task_id:
        raise HTTPException(status_code=500, detail="Failed to create task in database.")

    api_keys_to_pass = req.api_keys if req.api_keys is not None else {}
    configs_to_pass = req.configs if req.configs is not None else {}

    # 异步调用Celery任务
    process_video_task.delay(
        task_id=task_id,
        video_url=req.video_url,
        api_keys=api_keys_to_pass,
        configs=configs_to_pass
    )

    return {"task_id": task_id, "status": "pending"}

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """根据 task_id 查询任务的状态和结果。"""
    
    # 【优化】调用 database.py 中的函数来查询
    task = db_get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task

@app.post("/tasks/callback")
def task_callback(payload: CallbackPayload):
    """【内部接口】接收来自 Celery Worker 的回调通知，更新数据库。"""
    print(f"收到回调: task_id={payload.task_id}, status={payload.status}")
    
    # 【优化】调用 database.py 中的函数来更新
    if payload.status == "completed":
        update_task_as_completed(payload.task_id, payload.results)
    elif payload.status == "failed":
        error_message = payload.results.get("error", "Unknown error from worker.")
        update_task_as_failed(payload.task_id, error_message)
    
    return {"message": "Callback received and processed."}

# ==========================================
#      【新增】本地文件上传接口
# ==========================================
from fastapi import File, UploadFile
import shutil
import uuid

@app.post("/tasks/upload", response_model=TaskResponse, status_code=202)
def upload_file_task(
    file: UploadFile = File(...),
):
    """【新功能】直接上传本地视频文件进行处理"""
    
    # 1. 准备一个临时目录来存用户上传的视频
    upload_dir = os.path.abspath("temp_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # 2. 保存文件
    # 生成一个唯一文件名，防止重名冲突
    safe_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(upload_dir, safe_filename)
    
    print(f"接收到文件上传: {file.filename} -> 保存至 {file_path}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 3. 构造一个特殊的 URL，以 "file://" 开头，骗过系统
    # 这样 core_logic 就能识别这是个本地文件了
    local_video_url = f"file://{file_path}"
    
    # 4. 创建任务 (和之前的流程一样)
    task_id = db_create_task(local_video_url)
    
    # 5. 发送给 Celery (使用空的 api_keys 和 configs，让 worker 用默认值)
    process_video_task.delay(
        task_id=task_id,
        video_url=local_video_url,
        api_keys={},
        configs={}
    )
    
    return {"task_id": task_id, "status": "pending"}
