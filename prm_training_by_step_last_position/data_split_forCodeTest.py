import os
import json
import random
import shutil
from tqdm import tqdm

def split_jsonl_file(input_file, output_folder, train_ratio=0.8):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
        lines = []
        for line in tqdm(raw_lines, desc="Reading data"):
            temp_data = json.loads(line)
            if temp_data.get('label') is not None:
                if len(temp_data.get('label')) > 0:
                    # 查找<|reserved_special_token_250|>的个数
                    prompt = temp_data.get('prompt')
                    count = prompt.count('<|reserved_special_token_250|>')  
                    if count == len(temp_data.get('label')):
                        lines.append(line)

    # 打乱数据顺序
    random.shuffle(lines)

    # 分割数据
    train_data = lines[:100000]
    test_data = lines[100000:110000]

    # 输出文件路径
    train_file = os.path.join(output_folder, 'train.jsonl')
    test_file = os.path.join(output_folder, 'test.jsonl')

    # 写入 train.jsonl 文件
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line)

    # 写入 test.jsonl 文件
    with open(test_file, 'w', encoding='utf-8') as f:
        for line in test_data:
            f.write(line)

    print(f"Data has been split and saved to {output_folder}")

# 示例用法
input_file = 'F://code//github//math-feedback//math-feedback//prm_training_by_step_last_position_with_eval//raw_data//math-shepherd2.jsonl'  # 替换为你的 JSONL 文件路径
output_folder = os.path.join(os.path.dirname(input_file), 'test')

split_jsonl_file(input_file, output_folder)
