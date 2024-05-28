import json
import os
import uuid
import re
from collections import defaultdict

def read_and_separate_jsonl(folder_path, output_chinese, output_english):
    """读取文件夹中的所有jsonl文件，并根据prompt的语言分别保存。"""
    prompts_seen = set()

    # 打开两个输出文件准备写入
    with open(output_chinese, 'w', encoding='utf-8') as file_chinese, \
         open(output_english, 'w', encoding='utf-8') as file_english:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        entry = json.loads(line)
                        prompt = entry.get('prompt', '')

                        # 检查prompt是否已处理过，以去重
                        if prompt not in prompts_seen:
                            prompts_seen.add(prompt)
                            new_entry = transform_entry(entry)
                            
                            # 检查语言并写入相应的文件
                            if is_chinese(prompt):
                                json.dump(new_entry, file_chinese, ensure_ascii=False)
                                file_chinese.write('\n')
                            else:
                                json.dump(new_entry, file_english, ensure_ascii=False)
                                file_english.write('\n')

def is_chinese(text):
    """检查文本中是否含有中文字符。"""
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def transform_entry(entry):
    """转换条目格式，生成新的ID，并合并steps和step_labels。"""
    new_id = str(uuid.uuid4())  # 生成新的随机ID
    steps = entry.get('steps', [])
    step_labels = entry.get('step_labels', [])

    # 合并steps和step_labels
    solution = {}
    for i, (content, label) in enumerate(zip(steps, step_labels), 1):
        solution[f"Step {i}"] = {"content": content, "label": label}

    # 构建新的条目格式
    transformed = {
        "id": new_id,
        "prompt": entry.get('prompt', ''),
        "answer": entry.get('answer', ''),
        "solution": solution
    }
    return transformed

# 使用示例
folder_path = 'F://code//github//ChatGLM-MathV2//raw_data//math_chatglm_raw_data'
output_chinese = 'F://code//github//ChatGLM-MathV2//data//math_chatglm//chinese_prompts.jsonl'
output_english = 'F://code//github//ChatGLM-MathV2//data//math_chatglm//english_prompts.jsonl'
read_and_separate_jsonl(folder_path, output_chinese, output_english)
