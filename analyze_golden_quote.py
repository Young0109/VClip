from openai import OpenAI
import json
import os

api_client = None

def initialize_api_client(api_key: str):
    global api_client
    if api_client is None:
        api_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        print("API 客户端初始化完成。")

def analyze_golden_quote_with_explanation(text: str) -> dict:
    """使用 DeepSeek 分析文本中是否包含“金句”，并返回评分、内容和解释。"""
    if api_client is None:
        print(" API 客户端尚未初始化，请先调用 initialize_api_client()")
        return {"score": 0, "quote": "", "explanation": "API client not initialized."}

    if not text or not isinstance(text, str):
        return {"score": 0, "quote": "", "explanation": "Input text is empty."}

    system_prompt = """
你是一位经验丰富的病毒式内容总监和社交媒体专家。
你的任务是分析以下文本段落，判断其中是否包含“金句”。

“金句”指的是那些观点鲜明、高度凝练、富有哲理、情感强烈、易于引发共鸣或适合在社交媒体上传播的句子。

请严格按照以下 JSON 格式返回你的分析结果，不要包含任何额外的解释或文字。
返回一个 JSON 对象，包含三个键：
1. "score": 一个从 0 到 10 的整数，分数越高代表“金句”的潜质越大。如果完全没有，则为 0。
2. "quote": 具体的金句文本，如果没有则返回空字符串。
3. "explanation": 一段简洁的中文文字，解释你给出该分数的原因。如果不是金句，请说明理由；如果是，请解释它为什么有传播潜力。
"""
    try:
        response = api_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            max_tokens=512,
            temperature=0.2,
        )
        result_str = response.choices[0].message.content
        result_json = json.loads(result_str)
        
        return {
            "score": int(result_json.get("score", 0)),
            "quote": str(result_json.get("quote", "")),
            "explanation": str(result_json.get("explanation", "模型未提供解释。"))
        }

    except Exception as e:
        print(f" 金句检测 API 调用失败: {e}")
        return {"score": 0, "quote": "", "explanation": f"API Error: {e}"}

if __name__ == "__main__":
    my_api_key = "sk-984f91a660ca40ab9427e513a97f67ca" 
    
    initialize_api_client(my_api_key)

    paragraph_with_quote = "我们常说，技术的发展是一把双刃剑。但真正重要的，不是剑本身，而是握剑的人。这决定了我们走向哪个未来。"
    paragraph_long_and_mixed = "回顾我们这个季度的表现，财报数据显示用户增长了15%，这个数据符合团队的预期，基本上是按照项目计划书稳步推进的，但真正让我感触最深的，其实并不是这些冰冷的数字，而是我发现，一个团队真正的力量，并非来自完美的KPI指标，而是源于每个人心中那份不可替代的归属感与凝聚力，这才是我们能克服所有困难的基石。因此，在下个阶段的工作安排中，除了要继续优化A/B测试的流程，我们还需要安排一次团队建设活动。"

    if api_client:
        result_with = analyze_golden_quote_with_explanation(paragraph_with_quote)
        print(f"\n输入 (含金句): '{paragraph_with_quote}'")
        print(f"得分: {result_with['score']}, 金句内容: '{result_with['quote']}'")
        print(f"【推荐理由】: {result_with['explanation']}")
        print("-" * 20)
        
        result_long = analyze_golden_quote_with_explanation(paragraph_long_and_mixed)
        print(f"\n输入 (长句混合): '{paragraph_long_and_mixed}'")
        print(f"得分: {result_long['score']}, 金句内容: '{result_long['quote']}'")
        print(f"【推荐理由】: {result_long['explanation']}")
