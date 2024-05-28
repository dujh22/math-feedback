import json
import os
import re
import sys
from tqdm import tqdm
from utils.highlight_equations import highlight_equations


def split_response(response): # 使用正则表达式按换行符分割响应文本
    # 首先判断\n\n存在的数量，如果超过超过一个则按照这个划分
    if response.count('\n\n') >= 2:
        steps = re.split(r"\n\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps
    # 然后判断\n存在的数量，如果超过一个则按照这个划分
    if response.count('\n') >= 2:
        steps = re.split(r"\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps
    # 否则按照句号划分
    else:
        # 使用正则表达式按句号切割非小数点
        if '。' in response:
            steps = re.split(r'。', response)
        else:
            steps = re.split(r'(?<=[^.0-9])\.(?=[^0-9])', response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps

def process_json_line(line):
    # 加载原始JSON
    data = json.loads(line)

    # 初始化新的JSON格式
    if data.get("solution") is None:
        new_json = data
        new_json["solution"] = {}

        # 方案拆分
        split_responses = split_response(data['response'])

        # 处理每个解决方案部分
        for i, solution in enumerate(split_responses):
            new_json["solution"][f"Step {i+1}"] = {
                "content": solution.strip(),
            }

        # 处理每个解决方案部分的数学公式高亮
        for step, info in new_json["solution"].items():
            temp_content = info["content"]
            info["content"] = highlight_equations(temp_content)
        
    else: 
        # 如果已经存在solution字段，则直接使用，不需要二次拆分行
        new_json = data
        # 判断是否已经进行过公式高亮
        for step, info in new_json["solution"].items():
            # 如果该步骤不存在<<和>>则需要尝试高亮
            temp_content = info["content"]
            if '<<' not in temp_content and '>>' not in temp_content:
                info["content"] = highlight_equations(temp_content)
            
    # 返回新的JSON格式
    return json.dumps(new_json, ensure_ascii=False)

def process_files(source_folder, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 遍历文件夹中的所有JSONL文件
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".jsonl"):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_folder, file)
                
                with open(source_file_path, 'r', encoding='utf-8') as src_file, \
                     open(target_file_path, 'w', encoding='utf-8') as tgt_file:
                    for line in tqdm(src_file, desc='Processing'):
                        processed_line = process_json_line(line)
                        tgt_file.write(processed_line + '\n')

def create_jsonl_file(source_folder, data={}):
    # 确保源文件夹存在，如果不存在则创建
    if not os.path.exists(source_folder):
        os.makedirs(source_folder, exist_ok=True)
        
        # 完整的文件路径
        file_path = os.path.join(source_folder, 'api.jsonl')
        
        with open(file_path, 'w', encoding='utf-8') as file:
            if not data:
                # 提示用户输入每个 JSON 对象的数据
                question = input("请输入问题部分内容: ")
                solution = input("请输入解决方案部分内容: ")
                dataset = input("请输入数据集名称: ")
                
                # 创建 JSON 数据结构
                data = {
                    "question": question,
                    "solution": solution,
                    "dataset": dataset
                }
            else:
                # 将数据写入文件（每个 JSON 对象为一行）
                json.dump(data, file, ensure_ascii=False)
                file.write('\n')  # 换行，以便每个 JSON 对象占据一行
            
def Step1_SplitByRow(source_folder, target_folder, data = {}):
    create_jsonl_file(source_folder, data)
    process_files(source_folder, target_folder)

def main():
    if len(sys.argv) > 2:
        source_folder = sys.argv[1]
        target_folder = sys.argv[2]
    else:
        source_folder = 'F://code//github//ChatGLM-MathV2//data//test_data100//front'
        target_folder = 'F://code//github//ChatGLM-MathV2//data//test_data100//front_step1'

    Step1_SplitByRow(source_folder, target_folder)

if __name__ == '__main__':
    main()
