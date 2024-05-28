# 针对每一个step的content，判断是计算还是推理。如果是计算，应该会有基本的计算符号，比如+-*/=之类的。判断之后将结果保存为每一个Step对应json里面的一个新的键，就是is_calculation_or_reasoning，如果是计算则为1，是推理则为0

# 1.遍历源文件夹：读取每个JSONL文件。
# 2.修改JSON：对每个JSONL文件中的每一行JSON数据进行修改，根据content字段中是否包含计算符号（如+-*/=）来判断是计算还是推理，并添加新的键is_calculation_or_reasoning。
# 3.保存到目标文件夹：将修改后的JSONL保存到目标文件夹，保持原有的文件名和结构。

import os
import json
import re
import sys
from Step1_SplitByRow_forMathShepherd import Step1_SplitByRow_forMathShepherd
from utils.get_data_for_codeTest import get_data_for_codeTest
from tqdm import tqdm

def is_calculation(content):
    # 检查是否存在常见的计算符号
    if re.search(r'[\+\-\*/=%^]', content):
        return 1
    # 检查括号内是否存在计算符号，需要使用更复杂的正则表达式
    if re.search(r'\([^)]*[\+\-\*/=%^][^)]*\)', content):
        return 1
    return 0

def process_jsonl_file(source_path, dest_path):
    with open(source_path, 'r', encoding='utf-8') as src_file, \
         open(dest_path, 'w', encoding='utf-8') as dest_file:
        for line in tqdm(src_file, desc='Processing'):
            data = json.loads(line)
            # 遍历每一步的解决方案
            if 'solution' in data:
                for step, info in data['solution'].items(): # step变量会接收步骤的名称（如"Step 1"），而info变量会接收与这个步骤名称对应的字典值。
                    # 判断并添加新键
                    if info.get("equation") is not None:
                        info['is_calculation_or_reasoning'] = 1
                    else:
                        info['is_calculation_or_reasoning'] = is_calculation(info['content'])
            # 将修改后的数据写回新的JSONL文件
            json.dump(data, dest_file, ensure_ascii=False)
            dest_file.write('\n')

def Step2_IsCalculationOrReasoning(source_folder, target_folder):
    
    print("第二步判断单步是计算或推理……")

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jsonl'):
            source_path = os.path.join(source_folder, filename)
            print("正在处理文件:", source_path)
            dest_path = os.path.join(target_folder, filename)
            process_jsonl_file(source_path, dest_path)

# 使用方法：
def main2():
    code_test_state = True
    base_folder = "F://code//github//ChatGLM-MathV2"
    dataset_name = "peiyi9979_Math_Shepherd"
    source_folder = base_folder + '//raw_data//' + dataset_name
    if code_test_state:
        get_data_for_codeTest(source_folder)
        source_folder = source_folder + "_for_codeTest"
    mid_name = base_folder + '//data//' + dataset_name
    if code_test_state:
        target_folder1 = mid_name + "_for_codeTest" + "_Step1_SplitByRow_forMathShepherd"
        target_folder2 = mid_name + "_for_codeTest" + "_Step2_IsCalculationOrReasoning"
    else:
        target_folder1 = mid_name + "_Step1_SplitByRow_forMathShepherd"
        target_folder2 = mid_name + "_Step2_IsCalculationOrReasoning"

    Step1_SplitByRow_forMathShepherd(source_folder, target_folder1)
    Step2_IsCalculationOrReasoning(target_folder1, target_folder2)

def main():
    if len(sys.argv) > 2:
        source_folder = sys.argv[1]
        target_folder = sys.argv[2]
    else:
        source_folder = 'F://code//github//ChatGLM-MathV2//data//test_data100//front_step1'
        target_folder = 'F://code//github//ChatGLM-MathV2//data//test_data100//front_step2'
    Step2_IsCalculationOrReasoning(source_folder, target_folder)


if __name__ == '__main__':
    main()