import os
import json

# 从同一个文件夹内的 visual_scorer_api.py 导入API调用功能
from visual_scorer_api import call_qwen_vl

def generate_clip_explanation(clip_info: dict, frame_path: str, api_key: str) -> dict:
    print(f"  正在为片段 {clip_info['start']} -> {clip_info['end']} 生成智能解释...")
    
    text_summary = ""
    golden_quotes = []
    if clip_info.get('details', {}).get('text_info'):
        for info in clip_info['details']['text_info']:
            text_summary += info.get('text', '') + " "
            if info.get('golden_quote'):
                golden_quotes.append(info['golden_quote'])

    system_prompt = f"""
你是一位顶级的短视频内容策划。你的任务是基于我提供的视频截图和数据，为这个高光片段生成一份引人注目的推荐文案。

你需要输出一个JSON对象，包含三个键：
1. "title": 一个吸引人的、简短的视频标题 (不超过20个字)。
2. "tags": 一个包含3-5个相关关键词的JSON数组，用于社交媒体发布。
3. "reason": 一段话，生动地解释这个片段为什么精彩，为什么值得观看。

这是这个片段的数据：
- 视觉冲击力分数: {clip_info.get('details', {}).get('visual_score', 0):.2f}
- 文本内容平均分: {clip_info.get('details', {}).get('avg_text_score', 0):.2f}
- 综合推荐指数: {clip_info.get('total_score', 0):.2f}
- 对白摘要: "{text_summary[:100]}..."
- 可能的金句: {golden_quotes if golden_quotes else "无"}

请根据以上数据和这张视频截图进行创作。严格返回JSON格式。
"""

    try:
        response_text = call_qwen_vl(frame_path, system_prompt)
        json_str_match = response_text[response_text.find('{'):response_text.rfind('}')+1]
        if json_str_match:
            explanation_data = json.loads(json_str_match)
            return {
                "title": explanation_data.get("title", "AI生成标题"),
                "tags": explanation_data.get("tags", ["AI生成"]),
                "reason": explanation_data.get("reason", "该片段因其高综合评分而被AI选中。"),
                "score_details": clip_info.get("details", {})
            }
        else:
            raise ValueError("Response does not contain valid JSON.")

    except Exception as e:
        print(f"  -> 智能解释生成失败: {e}. 使用占位符内容。")
        return {
            "title": "高光片段",
            "tags": ["AI剪辑"],
            "reason": f"该片段因其高综合评分 ({clip_info.get('total_score', 0):.2f}) 被AI自动选中。",
            "score_details": clip_info.get("details", {})
        }
