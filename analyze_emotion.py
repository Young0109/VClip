import json
from openai import OpenAI 

def initialize_emotion_model():
    """这个函数现在只是一个占位符，因为我们已经切换到API模式。保留它是为了让 scoring_pipeline.py 的调用结构保持不变。"""
    print("情感分析模块已配置为API模式。")
    pass


def analyze_emotion(text: str, api_key: str) -> float:
    """ 分析单段文本的情感，它接收 api_key 并在内部创建客户端。返回一个从 -1.0 (非常消极) 到 +1.0 (非常积极) 的分数。"""
    if not text or not isinstance(text, str):
        return 0.0

    if not api_key or "sk-" not in api_key:
        print("错误：传入的 API Key 无效。")
        return 0.0

    # 在函数内部创建临时的API客户端
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        system_prompt = """
你是一位顶级的中文情感分析专家，擅长理解讽刺、委婉和复杂的语境。
你的任务是分析给定的文本，并给出一个介于 -1.0 到 1.0 之间的情感分数。
- 1.0 代表极端积极
- 0.0 代表中性
- -1.0 代表极端消极
请严格按照以下 JSON 格式返回你的分析结果，不要包含任何额外的解释或文字。
{"sentiment_score": 数值}
"""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            max_tokens=64,
            temperature=0.0,
        )
        result_str = response.choices[0].message.content
        result_json = json.loads(result_str)
        
        score = float(result_json.get("sentiment_score", 0.0))
        return max(-1.0, min(1.0, score))

    except Exception as e:
        print(f"情感分析API调用出错: {e}")
        return 0.0


if __name__ == "__main__":
    test_api_key = "YOUR_DEEPSEEK_API_KEY_HERE"
    
    if "YOUR_DEEPSEEK_API_KEY_HERE" not in test_api_key:
        text1 = "这部剧的男主角演技太神了，剧情也很紧凑，yyds！"
        text2 = "嗯，只能说我这种普通观众的审美水平还是有限，暂时还欣赏不来这么先锋的艺术"
        
        emotion_score1 = analyze_emotion(text1, test_api_key)
        emotion_score2 = analyze_emotion(text2, test_api_key)

        print("-" * 20)
        print(f"文本: '{text1}'\n情感分数: {emotion_score1:.4f}")
        print("-" * 20)
        print(f"文本: '{text2}'\n情感分数: {emotion_score2:.4f}")
    else:
        print("若要独立测试，请在此脚本的 if __name__ == '__main__' 中填入您的DeepSeek API Key。")
