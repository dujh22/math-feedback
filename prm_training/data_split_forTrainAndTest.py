import os
import json
import random
import shutil

def split_jsonl_file(input_file, output_folder, train_ratio=0.8):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 打乱数据顺序
    random.shuffle(lines)

    # 计算分割点
    split_point = int(len(lines) * train_ratio)

    # 分割数据
    train_data = lines[:split_point]
    test_data = lines[split_point:]

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
input_file = 'F://code//github//math-feedback//math-feedback//prm_training//raw_data//math-shepherd2.jsonl'  # 替换为你的 JSONL 文件路径
output_folder = os.path.join(os.path.dirname(input_file), 'math_shepherd2')

split_jsonl_file(input_file, output_folder)
