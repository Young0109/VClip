
import os
import json
from openai import OpenAI
import time
from opencc import OpenCC
import re 


def extract_json_from_response(text: str) -> str | None:
    """从可能含有额外文字的API响应中，智能地提取出最外层的JSON数组或对象。 """
   
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
        
    return None

def semantic_segment_final(transcript_path: str, output_path: str, api_key: str) -> bool:
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文稿文件 '{transcript_path}'")
        return False
    
    if not transcript_data:
        print("警告: 转写文稿为空，无法进行分段。")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return True

    cc = OpenCC('t2s')
    all_lines = [cc.convert(seg['text']).replace("`", "").replace("#", "") for seg in transcript_data]

    CHUNK_SIZE = 50
    all_segmentation_plans = []
    
    for i in range(0, len(all_lines), CHUNK_SIZE):
        chunk_lines = all_lines[i:i + CHUNK_SIZE]
        chunk_offset = i
        print(f"\n正在处理文本块 {i//CHUNK_SIZE + 1} (行号 {i+1} - {i+len(chunk_lines)})...")
        
        formatted_text = "".join([f"{j+1}: {line}\n" for j, line in enumerate(chunk_lines)])

        system_prompt = """
你是一个用于自动化流程的文本分段AI。你的唯一任务是将用户提供的、按行号标记的文稿，切分成在语义上连贯的段落。

**你必须严格遵守以下输出规则，否则整个程序将会失败：**
1.  你的回答**必须只包含一个JSON数组**，不能有任何其他文字、解释、或Markdown标记。
2.  JSON数组的格式必须是 `[{"start_line": number, "end_line": number}, ...]`。
3.  不要说“好的”或“这是您要的JSON”。你的回答的第一個字符必须是 `[`，最后一个字符必须是 `]`。

现在，请处理以下文稿：
"""
        
        llm_output_str = None
        for attempt in range(3):
            try:
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_text}],
                    # response_format={"type": "json_object"}, 
                    max_tokens=4096,
                    temperature=0.0, 
                )
                llm_output_str = response.choices[0].message.content
                
                extracted_json_str = extract_json_from_response(llm_output_str)
                
                if extracted_json_str:
                    json.loads(extracted_json_str) 
                    print("API 成功处理当前块，并成功提取JSON。")
                    llm_output_str = extracted_json_str 
                    break
                else:
                    print(f"警告：未能从API响应中提取出有效的JSON。响应内容: '{llm_output_str[:100]}...' (尝试次数: {attempt + 1}/3)")
                    llm_output_str = None 

            except Exception as e:
                print(f"警告: 调用或解析API时失败: {e} (尝试次数: {attempt + 1}/3)")
            
            if attempt < 2:
                time.sleep(3)
        
        if not llm_output_str:
            print(f"错误：块 {i//CHUNK_SIZE + 1} 经过多次尝试后处理失败。")
            continue
            
        chunk_plan = json.loads(llm_output_str)
        if isinstance(chunk_plan, dict):
            key_found = next((key for key in chunk_plan if isinstance(chunk_plan[key], list)), None)
            if key_found: chunk_plan = chunk_plan[key_found]
            else: chunk_plan = []
        for plan_item in chunk_plan:
            plan_item['start_line'] += chunk_offset
            plan_item['end_line'] += chunk_offset
            all_segmentation_plans.append(plan_item)
    
    if not all_segmentation_plans:
        return False 
        
    final_paragraphs = []
    for plan_item in all_segmentation_plans:
        start_line, end_line = plan_item['start_line'], plan_item['end_line']
        start_index, end_index = start_line - 1, end_line - 1
        
        if not (0 <= start_index < len(transcript_data) and 0 <= end_index < len(transcript_data)):
            continue
        
        paragraph_text = " ".join([transcript_data[i]['text'] for i in range(start_index, end_index + 1)])
        final_paragraphs.append({
            "start_time": transcript_data[start_index]['start_time'],
            "end_time": transcript_data[end_index]['end_time'],
            "paragraph_text": paragraph_text
        })

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_paragraphs, f, ensure_ascii=False, indent=4)
        
    print(f"\n全部完成！分段后的文稿已保存至: {output_path}")
    return True





if __name__ == "__main__":
    my_api_key = "sk-984f91a660ca40ab9427e513a97f67ca"   # <--- 可以改 API 密钥 

    google_drive_base_path = "/content/drive/My Drive/"    # <--- 改路径
    colab_output_folder = os.path.join(google_drive_base_path, "Colab_Output") 

    os.makedirs(colab_output_folder, exist_ok=True)

    input_transcript = os.path.join(colab_output_folder, "transcript.json") 
    output_segmented = os.path.join(colab_output_folder, "segmented_transcript.json")

    if os.path.exists(input_transcript):
        semantic_segment_final(
            transcript_path=input_transcript,
            output_path=output_segmented,
            api_key=my_api_key
        )
    else:
        print(f"错误: 找不到输入文件 '{input_transcript}'。请确保阶段一的脚本已成功运行，并且 'transcript.json' 文件存在于 Google Drive 的 '{colab_output_folder}' 路径下。")
