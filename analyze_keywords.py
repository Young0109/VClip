import jieba
from jieba import analyse
from openai import OpenAI
import json

jieba_initialized = False

def initialize_jieba_auto():
    global jieba_initialized
    if not jieba_initialized:
        jieba.initialize()
        jieba_initialized = True
        print(" jieba 初始化完成。")

def extract_keywords_candidates(text: str, top_k: int = 10) -> list:
    """[第一步：初筛] 使用 TF-IDF 提取候选关键词。我们提取更多候选（比如10个），给LLM更大的选择空间。"""
    if not text or not isinstance(text, str):
        return []
    
    allowed_pos = ('n', 'nr', 'ns', 'nt', 'nz', 'vn') # 使用词性过滤来提高候选词的质量
    keywords = analyse.extract_tags(text, topK=top_k, withWeight=False, allowPOS=allowed_pos)
    return keywords

def refine_keywords_with_llm(text: str, candidate_keywords: list, api_key: str) -> list:
    """[第二步：终审] 使用 LLM 从候选列表中精炼出最终的核心关键词。"""
    # 如果没有候选词，直接返回空列表
    if not candidate_keywords:
        return []
        
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    system_prompt = """
你是一位信息检索专家和高级编辑，你的任务是从给定的文本和候选关键词列表中，提炼出最核心、最不可或缺的概念。
这些词应该是这段话的“灵魂”，缺少了它们，这段话的核心意义就会丢失。
请严格按照 JSON 数组格式返回最终筛选出的关键词，不要包含任何额外的解释或文字。
例如: ["人工智能", "多模态"]
"""
    user_content = f"""
原始文本：
"{text}"

候选关键词列表：
{candidate_keywords}

请从中选出最核心的关键词：
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.0, 
        )
        # DeepSeek的JSON模式有时会返回一个带键的字典，需要做兼容处理
        result_str = response.choices[0].message.content
        result_data = json.loads(result_str)
        
        if isinstance(result_data, dict):
            for key in result_data:
                if isinstance(result_data[key], list):
                    return result_data[key]
            return [] 
        elif isinstance(result_data, list):
            return result_data
        else:
            return []

    except Exception as e:
        print(f" LLM 精炼关键词失败: {e}")
        return candidate_keywords 


if __name__ == "__main__":
    initialize_jieba_auto()

    my_api_key = "sk-984f91a660ca40ab9427e513a97f67ca"

    if my_api_key:
        paragraph_tech = "特斯拉的首席执行官马斯克宣布，他们将在火星上建立一个新的研发基地。"
        print(f"\n原始文本: '{paragraph_tech}'")
    
        # --- 第一步：TF-IDF 初筛 ---
        candidate_keywords = extract_keywords_candidates(paragraph_tech, top_k=10)
        print(f"\n[步骤1 - TF-IDF初筛] 候选关键词: {candidate_keywords}")

        # --- 第二步：LLM 终审 ---
        refined_keywords = refine_keywords_with_llm(paragraph_tech, candidate_keywords, my_api_key)
        print(f"\n[步骤2 - LLM终审] 精炼后核心词: {refined_keywords}")
    else:
        print("\n未提供 API Key。")
