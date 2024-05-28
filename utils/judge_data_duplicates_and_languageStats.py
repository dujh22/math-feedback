# 查重功能：遍历jsonl文件中的所有条目，比较它们的prompt字段，找出重复的条目。同时，我们也可以对同一文件夹下的所有jsonl文件进行查重。
# 语言分析功能：分析prompt中的字符，判断是中文还是英文。
# 两个文件夹的比较功能：比较两个文件夹中的jsonl文件集合的重复率。

import json
import os
import re
from collections import defaultdict # 用于计数

def read_jsonl(file_path):
    """读取jsonl文件，返回其中每个条目的prompt。"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data.get('prompt', ''))
    return prompts

def find_duplicates(prompts):
    """找出重复的prompt并返回它们。"""
    prompt_count = defaultdict(int)
    for prompt in prompts:
        prompt_count[prompt] += 1
    # return [prompt for prompt, count in prompt_count.items() if count > 1]
    prompt_count = {prompt: count for prompt, count in prompt_count.items() if count > 1}
    return prompt_count

def analyze_language(prompts):
    """分析prompt的语言类型（中文或英文）。"""
    language_stats = {'chinese': 0, 'english': 0}
    chinese_chars = re.compile(r'[\u4e00-\u9fff]+')
    for prompt in prompts:
        if chinese_chars.search(prompt):
            language_stats['chinese'] += 1
        else:
            language_stats['english'] += 1
    return language_stats

def process_folder(folder_path):
    """处理文件夹中的所有jsonl文件，查重并分析语言。"""
    all_prompts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(folder_path, file_name)
            prompts = read_jsonl(file_path)
            all_prompts.extend(prompts)
    
    duplicates = find_duplicates(all_prompts)
    language_stats = analyze_language(all_prompts)
    grada = len(duplicates) / len(all_prompts) * 100
    return duplicates, language_stats, grada

def find_common_prompts(prompts1, prompts2):
    """找出两个列表中共同的prompt。"""
    set1 = set(prompts1)
    set2 = set(prompts2)
    common_prompts = set1.intersection(set2)
    return common_prompts

def process_folder2(folder_path):
    """处理文件夹中的所有jsonl文件，提取prompts。"""
    all_prompts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(folder_path, file_name)
            prompts = read_jsonl(file_path)
            all_prompts.extend(prompts)
    return all_prompts

def compare_folders(folder1_path, folder2_path):
    """比较两个文件夹中的jsonl文件集合的重复率。"""
    prompts1 = process_folder2(folder1_path)
    prompts2 = process_folder2(folder2_path)
    common_prompts = find_common_prompts(prompts1, prompts2)
    duplicate_rate = len(common_prompts) / min(len(prompts1), len(prompts2))
    return duplicate_rate, common_prompts

folder_path1 = 'F://code//github//ChatGLM-MathV2//raw_data//math_chatglm_raw_data'
folder_path2 = 'F://code//github//ChatGLM-MathV2//raw_data//math_chatglm_raw_dataV2'
folder_path3 = 'F://code//github//ChatGLM-MathV2//data//math_chatglm'

# 使用示例1
duplicates, language_stats, grada = process_folder(folder_path3)
# print("Duplicates:", duplicates)
print("Duplicates:")
for prompt, count in duplicates.items():
    if count > 1:
        # print(f"Prompt: {prompt[0:5]}, Count: {count}")
        print(f"Prompt: {prompt}, Count: {count}")
print("Total Duplicates:", grada, "%")
print("Language Stats:", language_stats)

# 使用示例2
# duplicate_rate, common_prompts = compare_folders(folder_path1, folder_path2)
# print("Duplicate Rate:", duplicate_rate)
# # print("Common Prompts:", common_prompts)
