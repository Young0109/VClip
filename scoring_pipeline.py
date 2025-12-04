import os
import json

from analyze_emotion import initialize_emotion_model, analyze_emotion
from analyze_keywords import initialize_jieba_auto, extract_keywords_candidates, refine_keywords_with_llm
from analyze_golden_quote import initialize_api_client, analyze_golden_quote_with_explanation

def run_scoring_pipeline(input_path: str, output_path: str, api_key: str):
    """运行完整的文本评分流水线。"""
    
    # 1. 统一初始化所有需要的模型和客户端
    initialize_emotion_model()
    initialize_jieba_auto()
    initialize_api_client(api_key)

    # 2. 读取分段后的文稿
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            paragraphs = json.load(f)
        print(f"成功读取 {len(paragraphs)} 个段落，来自: {input_path}")
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_path}'")
        return

    scored_paragraphs = []
    # 3. 遍历每一个段落进行打分
    for i, p in enumerate(paragraphs):
        text = p['paragraph_text']
        print(f"\n[正在处理段落 {i+1}/{len(paragraphs)}]: '{text[:30]}...'")

        # 调用情感分析
        emotion_score = analyze_emotion(text, api_key)
        print(f"  - 情感分: {emotion_score:.4f}")

        # 调用关键词分析 (两步)
        candidate_keywords = extract_keywords_candidates(text)
        keywords = refine_keywords_with_llm(text, candidate_keywords, api_key)
        keyword_score = len(keywords)
        print(f"  - 关键词: {keywords} (得分: {keyword_score})")

        # 调用金句分析
        golden_quote_result = analyze_golden_quote_with_explanation(text)
        golden_quote_score = golden_quote_result['score']
        print(f"  - 金句分: {golden_quote_score}/10")

        # 4. 计算总分 (权重可以按需调整)
        final_score = (golden_quote_score * 1.0) + (emotion_score * 5) + (keyword_score * 1.5)

        # 5. 存储所有结果
        scored_paragraphs.append({
            "paragraph_index": i + 1,
            "start_time": p['start_time'],
            "end_time": p['end_time'],
            "paragraph_text": text,
            "analysis": {
                "emotion_score": round(emotion_score, 4),
                "keywords": keywords,
                "keyword_score": keyword_score,
                "golden_quote_score": golden_quote_score,
                "golden_quote_text": golden_quote_result['quote'],
                "golden_quote_explanation": golden_quote_result['explanation']
            },
            "final_text_score": round(final_score, 4)
        })

    # 6. 保存最终的带评分结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scored_paragraphs, f, ensure_ascii=False, indent=4)
        
    print(f"\n\n恭喜！带评分的文稿已保存至: {output_path} ")


if __name__ == "__main__":

    my_api_key = "sk-984f91a660ca40ab9427e513a97f67ca"

    if my_api_key:
        INPUT_JSON_PATH = "/content/drive/My Drive/Colab_Output/segmented_transcript.json"
        OUTPUT_JSON_PATH = "/content/drive/My Drive/Colab_Output/scored_transcript.json"

        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

        run_scoring_pipeline(
            input_path=INPUT_JSON_PATH,
            output_path=OUTPUT_JSON_PATH,
            api_key=my_api_key
        )
    else:
        print("\n无法执行评分流水线，因为未提供 API Key。")
