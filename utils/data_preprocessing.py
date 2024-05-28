import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import time
from Step1_SplitByRow_forMathShepherd import process_json_line, equation_extractor


# new_folder_suffix是输出文件的后缀
def data_preprocessing_for_math_shepherd(input_file_path, new_folder_suffix, num_points, language):
    # 1. 从大数据中获得小批量数据
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line) for line in file]

        # 2. 用于从math_shepherd数据集转化获得原始测试数据集
        lines2 = []
        line_prompt = set() # 用于查重
        chinese_chars = re.compile(r'[\u4e00-\u9fff]+')  # 用于判断是否为中文
        for data in lines:
            # 将每行的内容从JSON转换为字典
            input_text = data['input']        
            # 提取问题和解决方案
            split_point = input_text.find("Step 1:")
            question = input_text[:split_point].strip()
            solution = input_text[split_point:].strip()        
            # 移除所有的“Step n: ”和“ки”
            solution = solution.replace("ки", "")  # 删除所有的ки
            for i in range(1, 200):  # 假设步骤不超过200
                solution = solution.replace(f"Step {i}: ", "")       
            # 使用正则表达式移除<< >>和其内部的内容
            solution = re.sub(r'<<.*?>>', '', solution)
            # 更新字典
            new_data = {}
            new_data['question'] = question
            new_data['solution'] = solution
            new_data['standardLabelAnswer'] = json.loads(process_json_line(json.dumps(data)))

            # 3. 查重与语言分析
            if question not in line_prompt:
                if chinese_chars.search(question) and language == 'zn': # 如果是中文
                    lines2.append(new_data)
                    line_prompt.add(question)
                elif chinese_chars.search(question) == None and language == 'en': # 如果是英文
                    lines2.append(new_data)
                    line_prompt.add(question)
                elif language != 'zn' and language != 'en': # 如果不是中文也不是英文
                    lines2.append(new_data)
                    line_prompt.add(question)
            
            # 4. 限制数量
            if len(lines2) >= num_points:
                break
        
        # 5. 保存数据
        base_folder_path = os.path.dirname(input_file_path) # 确定新文件夹的路径
        new_folder_path = base_folder_path + '_' + new_folder_suffix
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        output_file_path = new_folder_path + '/' + new_folder_suffix + '.jsonl'
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line in lines2:
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')

# output_file_path是直接输出文件的路径
def data_preprocessing_for_math_shepherd2(input_file_path, output_file_path, num_points, language, has_label, has_response):
    # 1. 从大数据中获得小批量数据
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line) for line in file]

        # 2. 用于从math_shepherd数据集转化获得原始测试数据集
        lines2 = []
        line_prompt = set() # 用于查重
        chinese_chars = re.compile(r'[\u4e00-\u9fff]+')  # 用于判断是否为中文
        for data in lines:
            if has_label == 'hasnot':
                # 将每行的内容从JSON转换为字典
                input_text = data['input']        
                # 提取问题和解决方案
                split_point = input_text.find("Step 1:")
                question = input_text[:split_point].strip()
                # 更新字典
                new_data = {}
                new_data['question'] = question
                if has_response == 'has':
                    response = input_text[split_point:].strip()        
                    # 移除所有的“Step n: ”和“ки”
                    response = response.replace("ки", "")  # 删除所有的ки
                    for i in range(1, 1000):  # 假设步骤不超过1000
                        response = response.replace(f"Step {i}: ", "")       
                    # 使用正则表达式移除<< >>和其内部的内容
                    response = re.sub(r'<<.*?>>', '', response)
                    new_data['response'] = response
            elif has_label == 'hasset': # 如果有标签一定有response
                temp_new_data = json.loads(process_json_line(json.dumps(data)))
                new_data = {}
                new_data['question'] = temp_new_data['question']
                question = new_data['question']
                
                input_text = data['input']        
                split_point = input_text.find("Step 1:")
                response = input_text[split_point:].strip()        
                # 移除所有的“Step n: ”和“ки”
                response = response.replace("ки", "")  # 删除所有的ки
                for i in range(1, 1000):  # 假设步骤不超过1000
                    response = response.replace(f"Step {i}: ", "")       
                # 使用正则表达式移除<< >>和其内部的内容
                response = re.sub(r'<<.*?>>', '', response)
                new_data['response'] = response
                
                new_data['solution'] = temp_new_data['solution']
                new_data['dataset'] = temp_new_data['dataset']
                
            # 3. 查重与语言分析
            if question not in line_prompt:
                if chinese_chars.search(question) and language == 'zn': # 如果是中文
                    lines2.append(new_data)
                    line_prompt.add(question)
                elif chinese_chars.search(question) == None and language == 'en': # 如果是英文
                    lines2.append(new_data)
                    line_prompt.add(question)
                elif language != 'zn' and language != 'en': # 如果不是中文也不是英文
                    lines2.append(new_data)
                    line_prompt.add(question)
            
            # 4. 限制数量
            if len(lines2) >= num_points:
                break
        
        # 5. 保存数据
        _save_results(output_file_path, lines2)


def data_preprocessing_for_math_chatglm(input_file_path, output_file_path, num_points, language, has_label, has_response):
    # 1. 读取数据
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        chinese_chars = re.compile(r'[\u4e00-\u9fff]+')  # 用于判断是否为中文
        # 2. 处理数据
        lines2 = []
        line_prompt = set()
        for line in lines:
            data = json.loads(line)
            new_data = {}
            # 2.1 提取问题和解决方案
            if has_label == 'hasnot':
                new_data["question"] = data["prompt"].strip()
                if has_response == 'has':
                    response = data["answer"].strip().replace("\n\n\n", "\n")
                    new_data['response'] = response
            elif has_label == 'hasset':
                new_data["question"] = data["prompt"].strip()
                response = data["answer"].strip().replace("\n\n\n", "\n")
                new_data['response'] = response
                new_data['solution'] = {}
                for step, step_data in data["solution"].items():
                    equation = equation_extractor(step_data['content'])
                    new_data['solution'][step] = {
                        "content": step_data['content'],
                        "equation": equation,
                        "label": step_data['label']
                    }
            # 3. 查重与语言分析
            if new_data["question"] not in line_prompt:
                if chinese_chars.search(new_data["question"]) and language == 'zn':  # 如果是中文
                    lines2.append(new_data)
                    line_prompt.add(new_data["question"])
                elif chinese_chars.search(new_data["question"]) is None and language == 'en':  # 如果是英文
                    lines2.append(new_data)
                    line_prompt.add(new_data["question"])
                elif language != 'zn' and language != 'en':  # 如果不是中文也不是英文
                    lines2.append(new_data)
                    line_prompt.add(new_data["question"])
            # 4. 限制数量
            if len(lines2) >= num_points:
                break
            # 5. 保存数据
            _save_results(output_file_path, lines2)


def _save_results(_output_file_path, _datas):
    _base_folder_path = os.path.dirname(_output_file_path)  # 确定新文件夹的路径
    if not os.path.exists(_base_folder_path):
        os.makedirs(_base_folder_path)

    with open(_output_file_path, 'w', encoding='utf-8') as outfile:
        for line in _datas:
            json.dump(line, outfile, ensure_ascii=False)
            outfile.write('\n')

def main():
    # 请注意，目前数据预处理只支持math_shepherd数据集，其他数据集请参考并重新实现！！
    if len(sys.argv) > 7:
        # 输入文件路径
        input_file_path = sys.argv[1]
        # 输出文件路径后缀
        output_file_path = sys.argv[2]
        # 输出文件数量
        num_points = int(sys.argv[3])
        # 语言
        language = sys.argv[4]
        # 数据集
        dataset = sys.argv[5]
        # 是否有标签
        has_label = sys.argv[6]
        # 是否有回应
        has_response = sys.argv[7]
        if dataset == 'math_shepherd':
            data_preprocessing_for_math_shepherd2(input_file_path, output_file_path, num_points, language, has_label, has_response)
        elif dataset == 'math_chatglm':
            data_preprocessing_for_math_chatglm(input_file_path, output_file_path, num_points, language, has_label, has_response)
        else:
            raise ValueError(f"不支持{dataset}数据集，请注意，目前数据预处理只支持math_shepherd数据集，其他数据集请参考并重新实现！！")
            print("休眠1000秒...请中断自行退出")
            time.sleep(1000) # 休眠1000秒
    else:
        # 输入文件路径
        input_file_path = 'F://code//github//ChatGLM-MathV2//raw_data//peiyi9979_Math_Shepherd//math-shepherd.jsonl'
        # 输出文件路径后缀
        new_folder_suffix = 'math_shepherd_test_data100'
        # 输出文件数量
        num_points = 100
        # 语言
        language = 'en'
        # 数据集
        dataset = 'math_shepherd'
        # 调用函数
        if dataset == 'math_shepherd':
            data_preprocessing_for_math_shepherd(input_file_path, new_folder_suffix, num_points, language)
        else:
            print("请注意，目前数据预处理只支持math_shepherd数据集，其他数据集请参考并重新实现！！")
            print("休眠1000秒...请中断自行退出")
            time.sleep(1000) # 休眠1000秒

if __name__ == "__main__":
    main()
