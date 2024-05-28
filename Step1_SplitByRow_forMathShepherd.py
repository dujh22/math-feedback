# 第一步按行拆分

# 这个脚本会遍历指定的源文件夹，读取每一个JSONL文件，然后按照要求修改每一行JSON，并将修改后的JSONL保存到另一个目标文件夹中，保持原来的文件结构。

# 这个脚本的步骤如下：

# 1.遍历源文件夹：递归地遍历源文件夹中的所有文件和文件夹，找到所有的JSONL文件。
# 2.读取和处理JSONL文件：对于每一个JSON，读取其内容，然后按照要求修改。
# 3.保存修改后的文件：在目标文件夹中创建与原文件相同的路径，并保存修改后的JSONL。

import json
import os
import re
from tqdm import tqdm

def process_json_line(line):
    # 加载原始JSON
    data = json.loads(line)
    input_text = data['input']
    # input_text最后添加换行符
    if not input_text.endswith('\n'):
        input_text += '\n'
    label_text = data['label']
    # label_text最后添加换行符
    if not label_text.endswith('\n'):
        label_text += '\n'
    task = data['task']

    # 初始化新的JSON格式
    new_json = {
        "question": input_text.split('Step 1:')[0].strip(),
        "solution": {},
        "dataset": task
    }


    # 匹配所有步骤
    steps = re.findall(r"Step \d+:", input_text) # 返回一个包含所有匹配的列表。每个匹配项代表一个步骤的开始。
    # 遍历每个步骤
    for step in steps:
        step_number = step.strip(':') # 对于找到的每个步骤，例如"Step 1:"，这段代码去掉冒号（strip(':')），得到如"Step 1"这样的步骤编号。
        
        # 提取每一步的内容和标签
        content_pattern = re.escape(step) + r'(.*?)ки\n' # 用来在input_text中寻找特定步骤后到"ки\n"这个字符序列之前的所有字符
        label_pattern = re.escape(step) + r'(.*?)[\n]'
        
        content_match = re.search(content_pattern, input_text, re.DOTALL) # re.DOTALL参数使得.能匹配包括换行符在内的任何字符。
        label_match = re.search(label_pattern, label_text, re.DOTALL)

        if content_match and label_match:
            content = content_match.group(1).strip() # 如果成功找到内容和标签的匹配，group(1)用来提取括号()中匹配到的部分（即步骤的详细内容和标签文本）。
            label_raw = label_match.group(1).strip()
            label = 0 if label_raw.endswith('-') else 1 if label_raw.endswith('+') else label_raw

            # 识别公式
            equation = equation_extractor(content)
            # 更新solution字典
            new_json["solution"][step_number] = {
                "content": content,
                'equation': equation,
                "label": label
            }

    return json.dumps(new_json, ensure_ascii=False)


def equation_extractor(_content: str):
    equation = []
    pattern = r"<<(.+?)=(.+?)>>"
    matches = re.findall(pattern, _content)
    if len(matches) == 0:
        pattern = r"([\d\s\/*\-+.%]+)=([\d\s\/*\-+.%]+)"
        matches = re.findall(pattern, _content)
    for expr, expected_result in matches:
        expr = expr.lstrip("<")
        expr = expr.split("=")[0]
        expected_result = expected_result.rstrip(">")
        expected_result = expected_result.split("=")[-1]
        equation.append(f"{expr}={expected_result}")
    return equation


def Step1_SplitByRow_forMathShepherd(source_folder, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    print("第一步按行拆分……")

    # 遍历文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".jsonl"):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_folder, file)
                
                print("正在处理文件:", source_file_path)

                with open(source_file_path, 'r', encoding='utf-8') as src_file, open(target_file_path, 'w', encoding='utf-8') as tgt_file:
                    for line in tqdm(src_file, desc='Processing'):
                        processed_line = process_json_line(line)
                        tgt_file.write(processed_line + '\n')

def main():
    # 源文件夹和目标文件夹的路径
    # source_folder = 'F://code//github//ChatGLM-MathV2//raw_data//peiyi9979_Math_Shepherd'
    # target_folder = 'F://code//github//ChatGLM-MathV2//data//peiyi9979_Math_Shepherd_for_codeTest_step1'
    source_folder = 'F://code//github//ChatGLM-MathV2//raw_data//peiyi9979_Math_Shepherd'
    target_folder = 'F://code//github//ChatGLM-MathV2//raw_data//dujh22_Math_Shepherd'
    # source_folder = input("请输入源文件夹路径: ")
    # target_folder = input("请输入目标文件夹路径: ")
    Step1_SplitByRow_forMathShepherd(source_folder, target_folder)

if __name__ == '__main__':
    main()
