
import pymongo
from datetime import datetime, timezone
import uuid

# --- 数据库连接配置 ---
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "vclip_db"
TASK_COLLECTION_NAME = "tasks"

# --- 全局的数据库客户端实例 ---
try:
    client = pymongo.MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    client.admin.command('ismaster')
    db = client[DB_NAME]
    tasks_collection = db[TASK_COLLECTION_NAME]
    tasks_collection.create_index("task_id", unique=True)
    print("MongoDB 连接成功。")
except pymongo.errors.ConnectionFailure as e:
    print(f"错误：无法连接到 MongoDB。请确保 MongoDB 服务正在运行。错误信息: {e}")
    client = None
    tasks_collection = None

# --- 核心数据库操作函数 ---

def create_task(video_url: str) -> str | None:
    """在数据库中创建一个新的任务记录。"""
    # 【修正点】使用 is None 进行判断
    if tasks_collection is None:
        return None
        
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)
    
    task_document = {
        "task_id": task_id,
        "video_url": video_url,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "results": {"message": "任务已创建，正在等待处理..."}
    }
    
    tasks_collection.insert_one(task_document)
    print(f"数据库：已创建新任务，ID: {task_id}")
    return task_id

def get_task(task_id: str) -> dict | None:
    """根据 task_id 获取一个任务的信息。"""
    if tasks_collection is None:
        return None
    return tasks_collection.find_one({"task_id": task_id}, {'_id': 0})

def update_task_status(task_id: str, status: str, message: str = None):
    """更新任务的状态和消息。"""
    if tasks_collection is None:
        return
        
    update_fields = {"status": status, "updated_at": datetime.now(timezone.utc)}
    if message:
        update_fields["results.message"] = message
        
    tasks_collection.update_one({"task_id": task_id}, {"$set": update_fields})
    print(f"数据库：已更新任务 {task_id} 状态为 '{status}'")

def update_task_as_completed(task_id: str, final_results: dict):
    """将任务标记为“已完成”，并存入最终结果。"""
    if tasks_collection is None:
        return
        
    update_fields = {
        "status": "completed",
        "updated_at": datetime.now(timezone.utc),
        "results": final_results
    }
    tasks_collection.update_one({"task_id": task_id}, {"$set": update_fields})
    print(f"数据库：任务 {task_id} 已标记为“完成”。")

def update_task_as_failed(task_id: str, error_message: str):
    """将任务标记为“已失败”，并存入错误信息。"""
    if tasks_collection is None:
        return
        
    update_fields = {
        "status": "failed",
        "updated_at": datetime.now(timezone.utc),
        "results": {"message": "任务处理失败。", "error": error_message}
    }
    tasks_collection.update_one({"task_id": task_id}, {"$set": update_fields})
    print(f"数据库：任务 {task_id} 已标记为“失败”。")

def get_task_counts_by_status():
    """使用聚合管道按状态统计任务数量。"""
    if tasks_collection is None:
        print("错误：数据库未连接，无法执行聚合查询。")
        return {}
    
    print("数据库：正在执行按状态分组统计...")
    
    pipeline = [
        {
            "$group": {
                "_id": "$status",       # 按 status 字段的值进行分组
                "count": { "$sum": 1 }  # 对每个分组中的文档进行计数，结果存入 count 字段
            }
        }
    ]
    
    try:
        results = list(tasks_collection.aggregate(pipeline))
        # 将 MongoDB 返回的列表结果转换为更易于使用的字典
        # 例如： [{'id': 'completed', 'count': 58}, ...] -> {'completed': 58, ...}
        status_counts = {item['_id']: item['count'] for item in results}
        print(f"统计结果: {status_counts}")
        return status_counts
    except Exception as e:
        print(f"错误：聚合查询失败。错误信息: {e}")
        return {}
