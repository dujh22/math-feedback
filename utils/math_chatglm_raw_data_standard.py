import json
import os
import uuid
from collections import defaultdict

def read_and_merge_jsonl(folder_path):
    """读取文件夹中的所有jsonl文件，并合并内容，同时按prompt去重。"""
    prompts_seen = set()
    merged_data = []

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
                        merged_data.append(new_entry)
    
    return merged_data

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

def write_to_jsonl(data, output_file):
    # 如果文件夹不存在，则创建
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    """将合并后的数据写入新的jsonl文件。"""
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

# 把prompt是中文的和英文的单独进行保存

# 使用示例
folder_path = 'F://code//github//ChatGLM-MathV2//raw_data//math_chatglm_raw_data'
output_file = 'F://code//github//ChatGLM-MathV2//data//math_chatglm//math_chatglm_raw_datamerged_output.jsonl'
merged_data = read_and_merge_jsonl(folder_path)
write_to_jsonl(data=merged_data, output_file=output_file)
