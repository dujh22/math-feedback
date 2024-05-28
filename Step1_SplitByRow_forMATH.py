# 第一步按行拆分

# 这个脚本会遍历指定的源文件夹，读取每一个JSON文件，然后按照要求处理solution字段，并将修改后的JSON保存到另一个目标文件夹中，保持原来的文件结构。

# 这个脚本的步骤如下：

# 1.遍历源文件夹：递归地遍历源文件夹中的所有文件和文件夹，找到所有的JSON文件。
# 2.读取和处理JSON文件：对于每一个找到的JSON文件，读取其内容，然后按照要求修改solution字段。
# 3.保存修改后的文件：在目标文件夹中创建与原文件相同的路径，并保存修改后的JSON。

import os
import json
import re

def process_solution2(solution):
    # 根据\n分割字符串
    lines = solution.split('\n')
    return {str(index): line for index, line in enumerate(lines)}

def process_solution3(solution):
    # 根据\n分割字符串得到每行
    lines = solution.split('\n')
    
    # 定义一个用于分割句子的正则表达式，不会在小数点处分割
    sentence_splitter = r'\.(?!\d)'
    
    # 对每行进行处理，进一步按句号分割，但不分割小数点
    processed_lines = {}
    for index, line in enumerate(lines):
        # 使用 re.split 来安全地分割每一行，保留分割符号
        sentences = re.split(sentence_splitter, line)
        # 去除空字符串并保留结果
        processed_sentences = [sentence.strip() + '.' for sentence in sentences if sentence.strip()]
        processed_lines[str(index)] = processed_sentences[:-1] + [processed_sentences[-1][:-1]] if processed_sentences else []

    return processed_lines

def process_solution(text):
    # 预处理：替换掉 LaTeX 特殊命令和环境
    def preprocess_latex(text):
        # 匹配 \command{...}，包括没有 \end{...} 的 \begin{...}
        commands = r"\\(?:begin|item|section|subsection|subsubsection|paragraph|subparagraph|part|chapter|emph|textbf|textit|underline){[^}]*}"

        # 匹配 \command 无花括号的命令，如 \item 或 \bigskip
        simple_commands = r"\\(?:item|bigskip|medskip|smallskip|newpage|newline|linebreak)[^a-zA-Z]"

        # 为所有匹配到的命令插入分隔标记
        text = re.sub(commands, lambda x: f".\n{x.group(0)}.\n", text)
        text = re.sub(simple_commands, lambda x: f".\n{x.group(0)}\n", text)

        return text

    text = preprocess_latex(text)
    
    # 使用换行符和非小数点的句号进行分割
    sentences = re.split(r'(?<!\d)\.(?!\d)|\n', text)
    
    # 清理句子：移除多余空格
    final_sentences = [s.strip() for s in sentences if s.strip()]

    # 构造序号与句子的字典
    result = {i + 1: final_sentences[i] for i in range(len(final_sentences))}

    # 转换为 JSON 字符串
    return json.dumps(result, ensure_ascii=False, indent=4)

def process_json_file(source_path, target_path):
    with open(source_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 处理solution字段
        data['solution'] = process_solution(data['solution'])
    # 确保目标路径存在
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_all_json_files(source_dir, target_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.json'):
                source_path = os.path.join(root, file)
                # 构建目标文件的路径
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, relative_path)
                process_json_file(source_path, target_path)

# 源文件夹和目标文件夹的路径
source_dir = 'F://code//github//ChatGLM-MathV2//raw_data//MATH//train_temp'
target_dir = 'F://code//github//ChatGLM-MathV2//data//MATH//train_temp_step1'

# 开始处理
process_all_json_files(source_dir, target_dir)
