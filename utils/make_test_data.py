import json
import os
import re

def process_jsonl_file(input_file_path, output_file_path):
    '''
        该函数用于处理JSONL格式的文件，将输入文件中的内容转换为字典，提取问题和解决方案，并将其写入到输出文件中。它还会删除所有的“Step n: ”和“ки”，以及使用正则表达式移除<< >>和其内部的内容。
        用于从math_shepherd数据集转化获得原始测试数据集
    '''
    # 判断输出文件是否存在，如果不存在则需要创建
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    # 打开输入文件和输出文件
    with open(input_file_path, 'r', encoding='utf-8') as file, open(output_file_path, 'w', encoding='utf-8') as outfile:
        # 逐行读取文件
        for line in file:
            # 将每行的内容从JSON转换为字典
            data = json.loads(line)
            input_text = data['input']
            
            # 提取问题和解决方案
            split_point = input_text.find("Step 1:")
            question = input_text[:split_point].strip()
            solution = input_text[split_point:].strip()
            
            # 移除所有的“Step n: ”和“ки”
            solution = solution.replace("ки", "")  # 删除所有的ки
            for i in range(1, 20):  # 假设步骤不超过20
                solution = solution.replace(f"Step {i}: ", "")
            
            # 使用正则表达式移除<< >>和其内部的内容
            solution = re.sub(r'<<.*?>>', '', solution)

            # 更新字典
            new_data = {}
            new_data['question'] = question
            new_data['solution'] = solution
            
            # 将更新后的字典转换为JSON格式并写入文件
            json.dump(new_data, outfile, ensure_ascii=False)
            outfile.write('\n')  # 确保每个条目在新的一行

# 指定输入和输出文件路径
input_file_path = 'F://code//github//ChatGLM-MathV2//raw_data//peiyi9979_Math_Shepherd_for_codeTest\math-shepherd1-100.jsonl'
output_file_path = 'F://code//github//ChatGLM-MathV2//data//test_data100//test_data100.jsonl'

# 调用函数
process_jsonl_file(input_file_path, output_file_path)
