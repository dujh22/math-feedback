import json
import os
from tqdm import tqdm
from Step3_JudgmentStepCalculatedCorrectly import replace_calculated_result, llm_response

def read_jsonl(file_path):
    """读取JSONL文件，返回一个包含多个JSON对象的列表，并为每个对象添加一个唯一的索引作为ID。"""
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            entry = json.loads(line)
            entry['id'] = index  # 增加唯一标识符
            data.append(entry)
    return data

def read_processed_jsonl(file_path):
    """读取JSONL文件，返回一个包含多个JSON对象的列表，并为每个对象添加一个唯一的索引作为ID。"""
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            data.append(entry)
    return data


def append_jsonl(data, file_path):
    """追加数据到JSONL文件中，并确保目录存在。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')

def analyze_data(json_data, processed_json_data, output_file_path):
    """分析JSON对象列表，计算所需的统计数据。"""
    processed_ids = {entry['id'] for entry in processed_json_data}  # 创建一个包含所有已处理ID的集合

    # json_data = json_data[:20]
    total_entries = len(json_data)  # 总的JSON对象数
    
    # 用于存储正确判断的样例数
    correct_judgments_by_case = {
        'only_calculation_cases': 0, # 统计只有计算步骤的样例数
        'calculation_and_reasoning_cases': 0, # 统计既有计算步骤又有推理步骤的样例数
        'only_reasoning_cases': 0, # 统计只有推理步骤的样例数

        'LabelJudgmentCaseCalculatedCorrectly': 0, # 统计计算步骤判断正确的样例数（依据人工标签判断）
        'LabelJudgmentCaseEquationCorrectly': 0, # 统计计算公式判断正确的样例数（依据人工标签判断）
        'LabelJudgmentCaseReasoningCorrectly': 0, # 统计推理步骤判断正确的样例数（依据人工标签判断）

        'LLMJudgmentCaseCalculatedCorrectly': 0, # 统计计算步骤判断正确的样例数（依据LLM判断）
        'LLMJudgmentCaseEquationCorrectly': 0, # 统计计算公式判断正确的样例数（依据LLM判断）
        'LLMJudgmentCaseReasoningCorrectly': 0, # 统计推理步骤判断正确的样例数（依据LLM判断）

        'correct_label_cases': 0, # 正确样例数
        'correct_explain_cases': 0, # 正确解释数
        'total_cases': 0,  # 总样例数

        'sympy_count':0,  # 使用SymPy的次数
        'python_code_count':0  # 使用Python编程的次数
    }
    # 用于存储正确判断的步骤数
    correct_judgments_by_step = {
        'calculation_steps': 0, # 统计计算步骤数
        'reasoning_steps': 0, # 统计推理步骤数

        'LabelJudgmentStepCalculatedCorrectly': 0, # 统计计算步骤判断正确的步骤数（依据人工标签判断）
        'LabelJudgmentStepEquationCorrectly': 0, # 统计计算公式判断正确的步骤数（依据人工标签判断）
        'LabelJudgmentStepReasoningCorrectly': 0, # 统计推理步骤判断正确的步骤数（依据人工标签判断）

        'LLMJudgmentCaseCalculatedCorrectly': 0, # 统计计算步骤判断正确的步骤数（依据LLM判断）
        'LLMJudgmentCaseEquationCorrectly': 0, # 统计计算公式判断正确的步骤数（依据LLM判断）
        'LLMJudgmentCaseReasoningCorrectly': 0, # 统计推理步骤判断正确的步骤数（依据LLM判断）        

        'correct_label_steps': 0, # 正确步骤数
        'correct_explain_steps': 0, # 正确解释数
        'total_steps': 0,  # 总步骤数

        'sympy_count':0,  # 使用SymPy的次数
        'python_code_count':0  # 使用Python编程的次数
    }

    # 针对已经处理过的样本点:
    for entry in tqdm(processed_json_data, desc='Processing'):
        correct_judgments_by_case['total_cases'] += 1
        
        total_label_correct = True
        total_explain_correct = True
        
        llm_total_calculate_correct = True
        llm_total_equation_correct = True
        llm_total_reasoning_correct = True
    
        label_total_calculate_correct = True
        label_total_equation_correct = True
        label_total_reasoning_correct = True

        total_python_used = False
        total_sympy_used = False 
        has_calculation = False
        has_reasoning = False      

        for step_key, step_info in entry['solution'].items():
            python_used = False
            sympy_used = False
            explain_correct = True

            # 计算总步骤数
            correct_judgments_by_step['total_steps'] += 1
            
            # 检查每种判断类型
            if step_info['is_calculation_or_reasoning'] == 1:
                correct_judgments_by_step['calculation_steps'] += 1
                has_calculation = True
                Flag = False
                # 检测计算步的标签是否正确
                if step_info['label'] == 1:
                    if all(temp == 1 for temp in step_info['JudgmentStepCalculatedCorrectly']):
                        correct_judgments_by_step['LabelJudgmentStepCalculatedCorrectly'] += 1
                        Flag = True
                    else:
                        label_total_calculate_correct = False

                    if all(temp == 1 for temp in step_info['JudgmentStepEquationCorrectly']):
                        correct_judgments_by_step['LabelJudgmentStepEquationCorrectly'] += 1
                        Flag = True
                    else:
                        label_total_equation_correct = False

                    if Flag == True:
                        correct_judgments_by_step['correct_label_steps'] += 1
                    else:
                        total_label_correct = False
                    
                else:
                    if any(temp == 0 for temp in step_info['JudgmentStepCalculatedCorrectly']):
                        correct_judgments_by_step['LabelJudgmentStepCalculatedCorrectly'] += 1
                        Flag = True
                    else:
                        label_total_calculate_correct = False

                    if any(temp == 0 for temp in step_info['JudgmentStepEquationCorrectly']):
                        correct_judgments_by_step['LabelJudgmentStepEquationCorrectly'] += 1
                        Flag = True
                    else:
                        label_total_equation_correct = False

                    if Flag == True:
                        correct_judgments_by_step['correct_label_steps'] += 1
                    else:
                        total_label_correct = False

                # 检测LLM针对计算步骤的判断是否正确
                response1 = step_info['LLMJudgmentStepCalculatedCorrectly']
                # 获取response的第一句话或者如果没有符号就是完整的response
                response1_first_sentence = response1.split(".")[0]
                if response1_first_sentence[:3].lower() == "yes" or "yes" in response1_first_sentence.lower():  # 如果计算正确
                    correct_judgments_by_step['LLMJudgmentStepCalculatedCorrectly'] += 1
                else:
                    llm_total_calculate_correct = False
                    explain_correct = False


                # 检测LLM针对计算公式的判断是否正确
                response1 = step_info['LLMJudgmentStepEquationCorrectly']
                # 获取response的第一句话或者如果没有符号就是完整的response
                response1_first_sentence = response1.split(".")[0]
                if response1_first_sentence[:3].lower() == "yes" or "yes" in response1_first_sentence.lower():  # 如果计算正确
                    correct_judgments_by_step['LLMJudgmentStepEquationCorrectly'] += 1
                else:
                    llm_total_equation_correct = False
                    explain_correct = False

            else:

                correct_judgments_by_step['reasoning_steps'] += 1
                has_reasoning = True

                if step_info['label'] == 1:
                    if step_info['JudgmentStepReasoningCorrectly'] == 1:
                        correct_judgments_by_step['LabelJudgmentStepReasoningCorrectly'] += 1
                        correct_judgments_by_step['correct_label_steps'] += 1
                    else:
                        label_total_reasoning_correct = False
                else:
                    if step_info['JudgmentStepReasoningCorrectly'] == 0:
                        correct_judgments_by_step['LabelJudgmentStepReasoningCorrectly'] += 1
                        correct_judgments_by_step['correct_label_steps'] += 1
                    else:
                        label_total_reasoning_correct = False


                # 检测针对推理步骤的判断是否正确
                response2 = step_info['LLMJudgmentStepReasoningCorrectly']
                # 获取response的第一句话或者如果没有符号就是完整的response
                response2_first_sentence = response2.split(".")[0]
                if response2_first_sentence[:3].lower() == "yes" or "yes" in response2_first_sentence.lower():  # 如果推理正确
                    correct_judgments_by_step['JudgmentStepReasoningCorrectly'] += 1
                else:
                    llm_total_reasoning_correct = False
                    explain_correct = False
            
            if explain_correct == True:
                correct_judgments_by_step['correct_explain_steps'] += 1
            else:
                total_explain_correct = False

            # 检查是否使用了SymPy或Python编程  
            if 'sympy and llm' in step_info['leftSideOfEqual_use_sympy_or_llm']:
                python_used = True
                total_python_used = True
            elif 'sympy' in step_info['leftSideOfEqual_use_sympy_or_llm']:
                sympy_used = True
                total_sympy_used = True
            if 'sympy and llm' in step_info['rightSideOfEqual_use_sympy_or_llm']:
                python_used = True
                total_python_used = True
            elif 'sympy' in step_info['rightSideOfEqual_use_sympy_or_llm']:
                sympy_used = True
                total_sympy_used = True
        
            if sympy_used:
                correct_judgments_by_step['sympy_count'] += 1
            if python_used:
                correct_judgments_by_step['python_code_count'] += 1

        if has_calculation and has_reasoning:
            correct_judgments_by_case['calculation_and_reasoning_cases'] += 1
        elif has_calculation:
            correct_judgments_by_case['only_calculation_cases'] += 1
        else:
            correct_judgments_by_case['only_reasoning_cases'] += 1

        if total_label_correct == True:
            correct_judgments_by_case['correct_label_cases'] += 1
        if total_explain_correct == True:
            correct_judgments_by_case['correct_explain_cases'] += 1

        if label_total_calculate_correct == True:
            correct_judgments_by_case['LabelJudgmentStepCalculatedCorrectly'] += 1
        if label_total_equation_correct == True:
            correct_judgments_by_case['LabelJudgmentStepEquationCorrectly'] += 1
        if label_total_reasoning_correct == True:
            correct_judgments_by_case['LabelJudgmentStepReasoningCorrectly'] += 1

        if llm_total_calculate_correct == True:
            correct_judgments_by_case['LLMJudgmentStepCalculatedCorrectly'] += 1
        if llm_total_equation_correct == True:
            correct_judgments_by_case['LLMJudgmentStepEquationCorrectly'] += 1
        if llm_total_reasoning_correct == True:
            correct_judgments_by_case['LLMJudgmentStepReasoningCorrectly'] += 1
        
        if total_python_used == True:
            correct_judgments_by_case['python_code_count'] += 1
        if total_sympy_used == True:
            correct_judgments_by_case['sympy_count'] += 1



    # for entry in json_data:
    for entry2 in tqdm(json_data, desc='Processing'):
        if entry2['id'] in processed_ids:
            continue  # 跳过已处理的数据

        all_correct = True
        sympy_used = False
        python_used = False
        
        history2 = ""
        for step_key2, step_info2 in entry2['solution'].items():
            # 计算总步骤数
            correct_judgments['total_steps'] += 1
            
            # 检查每种判断类型

            # 检测针对计算步骤的判断是否正确

            # 获取修正后的结果
            temp_content2 = replace_calculated_result(step_info2['content'], step_info2["equation"], step_info2["JudgmentStepCalculatedCorrectly"], step_info2["StepCalculatedCorrectlyResult"])
            temp_true2 = False
            if all(item == 1 for item in step_info2['JudgmentStepCalculatedCorrectly']):
                temp_true2 = True
            
            if temp_true2 == True:
                # prompt1 = f"""我正在尝试检查一个数学问题的求解过程是否计算正确、推理合理。具体问题是：{entry2['question']}。\n\n 我目前采用的解题步骤如下：{history2} \n\n 现在我要检查的步骤是：{step_key2}，内容是：{step_info2['content']}。\n\n 我认为计算是正确的。\n\n 请问我的判断是否正确？（是/否）"""
                prompt1 = f"""I am trying to check that the solution to a math problem is computationally correct and reasoned correctly. The specific problem is: {entry2['question']} \n\n The steps I have used so far to solve the problem are as follows:{history2} \n\n The steps I would like to check now are:{step_key2} and the content is:{step_info2['content']}. \n\n I think the calculation is correct. \n\n May I ask if my judgment is correct? (Yes/No)"""
            else:
                # prompt1 = f"""我正在尝试检查一个数学问题的求解过程是否计算正确、推理合理。具体问题是：{entry2['question']}。\n\n 我目前采用的解题步骤如下：{history2} \n\n 现在我要检查的步骤是：{step_key2}，内容是：{step_info2['content']}。\n\n 我认为这一步计算是错误的，应该修改为：{temp_content2} 请问我的判断和修改是否正确？\n\n（是/否）"""
                prompt1 = f"""I'm trying to check that the solution to a math problem is computationally correct and reasoned correctly. The specific problem is: {entry2['question']} \n\n The solution steps I have used so far are as follows:{history2} \n\n Now the steps I want to check are:{step_key2} and the content is:{step_info2['content']}. \n\n I think this step is calculated incorrectly and should be modified as: {temp_content2} Am I correct in my judgment and modification? \n\n (yes/no)"""

            for i in range(10):
                try:
                    response1 = llm_response(prompt1, use_glm_or_gpt='gpt')
                    break
                except:
                    response1 = "no llm judge"
            step_info2['LLMJudgmentStepCalculatedCorrectly'] = response1  # 更新entry

            if response1 == "no llm judge":
                print(response1)
            else:
                # 获取response的第一句话或者如果没有符号就是完整的response
                response1_first_sentence = response1.split(".")[0]

                if response1_first_sentence[:3].lower() == "yes" or ("correct" in response1_first_sentence.lower() and "incorrect" not in response1_first_sentence.lower()):  # 如果计算正确
                    correct_judgments['JudgmentStepCalculatedCorrectly'] += 1
                else:
                    all_correct = False
            
            # 检测针对推理步骤的判断是否正确
            temp_true1 = False
            if step_info2['JudgmentStepReasoningCorrectly'] == 0:
                temp_true1 = True
            
            if temp_true1 == True:
                # prompt2 = f"""我正在尝试检查一个数学问题的求解过程是否计算正确、推理合理。具体问题是：{entry2['question']}。\n\n 我目前采用的解题步骤如下：{history2} \n\n 现在我要检查的步骤是：{step_key2}，内容是：{step_info2['content']}。\n\n 我认为这一步的推理是正确的。\n\n 请问我的判断是否正确？（是/否）"""
                prompt2 = f"""I am trying to check that the solution to a math problem is computationally correct and reasoned correctly. The specific problem is: {entry2['question']} \n\n The solution steps I have used so far are as follows:{history2} \n\n Now the steps I want to check are:{step_key2} and the content is:{step_info2['content']}. \n\n I think the reasoning in this step is correct. \n\n May I ask if my judgment is correct? (Yes/No)"""
            else:
                # prompt2 = f"""我正在尝试检查一个数学问题的求解过程是否计算正确、推理合理。具体问题是：{entry2['question']}。\n\n 我目前采用的解题步骤如下：{history2} \n\n 现在我要检查的步骤是：{step_key2}，内容是：{step_info2['content']}。\n\n 我认为这一步的推理是错误的，应该修改为：{step_info2['StepReasoningCorrectlyResult']}。\n\n 请问我的判断是否正确？（是/否）"""
                prompt2 = f"""I'm trying to check that the solution to a math problem is computationally correct and reasoned correctly. The specific problem is: {entry2['question']} \n\n The solution steps I have used so far are as follows:{history2} \n\n Now the steps I want to check are:{step_key2} and the content is:{step_info2['content']}. \n\n I think the reasoning in this step is wrong and should be changed to: {step_info2['StepReasoningCorrectlyResult']}. \n\n Is my judgment correct? (Yes/No)"""
            
            for i in range(10):
                try:
                    response2 = llm_response(prompt2, use_glm_or_gpt='gpt')
                    break
                except:
                    response2 = "no llm judge"
            step_info2['LLMJudgmentStepReasoningCorrectly'] = response2  # 更新entry

            if response2 == "no llm judge":
                print(response2)
            else:
                # 获取response的第一句话或者如果没有符号就是完整的response
                response2_first_sentence = response2.split(".")[0]

                if response2_first_sentence[:3].lower() == "yes" or ("correct" in response2_first_sentence.lower() and "incorrect" not in response2_first_sentence.lower()):  # 如果推理正确
                    correct_judgments['JudgmentStepReasoningCorrectly'] += 1
                else:
                    all_correct = False
    
            # 检查是否使用了SymPy或Python编程
            if 'leftSideOfEqual_use_sympy_or_llm' in step_info2:
                if 'sympy and llm' in step_info2['leftSideOfEqual_use_sympy_or_llm']:
                    python_used = True
                else:
                    sympy_used = True
            if 'rightSideOfEqual_use_sympy_or_llm' in step_info2:
                if 'sympy and llm' in step_info2['rightSideOfEqual_use_sympy_or_llm']:
                    python_used = True
                else:
                    sympy_used = True
            
            history2 += f"{step_key2}: {step_info2['content']}\n"

        # 处理完毕后，将数据追加到新文件
        append_jsonl(entry2, output_file_path)
        
        if all_correct:
            all_correct_json_count += 1
        if sympy_used:
            sympy_count += 1
        if python_used:
            python_code_count += 1
    
    return {
        'correct_judgments': correct_judgments,
        'all_correct_json_count': all_correct_json_count,
        'sympy_count': sympy_count,
        'python_code_count': python_code_count,
        'total_entries': total_entries
    }

def get_display_width(text):
    """计算字符串的显示宽度，中文字符计为2，英文字符计为1"""
    width = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            width += 2  # 假设中文字符宽度为2
        else:
            width += 1
    return width

def print_padded_line(label, value, total_width=40):
    """打印填充后的行，保证右侧对齐"""
    label_width = get_display_width(label)
    spaces = ' ' * (total_width - label_width - len(value))
    print(f"{label}{spaces}{value}")

def print_statistics(stats):
    """打印统计信息为表格形式，并格式化为固定宽度的列，使用print函数，并考虑字符宽度。"""
    print_padded_line("统计指标", "值")
    print('-' * 40)  # 根据总宽度调整分隔线长度

    # 打印数据行
    print_padded_line("计算步骤正确性准确率", f"{stats['correct_judgments']['JudgmentStepCalculatedCorrectly'] / stats['correct_judgments']['total_steps'] * 100:.2f}%")
    print_padded_line("推理步骤正确性准确率", f"{stats['correct_judgments']['JudgmentStepReasoningCorrectly'] / stats['correct_judgments']['total_steps'] * 100:.2f}%")
    print_padded_line("全部正确的JSON占比", f"{stats['all_correct_json_count'] / stats['total_entries'] * 100:.2f}%")
    print_padded_line("使用SymPy的占比", f"{stats['sympy_count'] / stats['total_entries'] * 100:.2f}%")
    print_padded_line("使用Python编程的占比", f"{stats['python_code_count'] / stats['total_entries'] * 100:.2f}%")

def Check2_CalculateAccuracy(input_file_path):
    # 根据需要修改文件路径

    # 检查filename中"Step"的位置并插入"Check2"
    output_file_path = input_file_path.replace("Step4", "Check2Step4")
    
    processed_data = read_processed_jsonl(output_file_path)
    json_data = read_jsonl(input_file_path)
    stats = analyze_data(json_data, processed_data, output_file_path)
    
    # 打印统计信息
    print_statistics(stats)

def main():
    input_file_path  = 'F://code//github//ChatGLM-MathV2//data//peiyi9979_Math_Shepherd_for_codeTest_Step4_JudgmentStepReasoningCorrectly//math-shepherd.jsonl_1-100.jsonl'
    Check2_CalculateAccuracy(input_file_path)

if __name__ == "__main__":
    main()
