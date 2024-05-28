import json
import os
import csv
from tqdm import tqdm
from Step3_JudgmentStepCalculatedCorrectly import replace_calculated_result, llm_response, TGI_URL, CRITIC_URL

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

def llm_judge_response(question, step, info, type, backbone = "chatglm_platform", url = CRITIC_URL):
    history = ""
    correct_content = "" 
    for temp_step, his in info['history_json'].items():
        if temp_step != step:
            history += f"{step}: {his}\n"
        else:
            correct_content = his

    content = info['content']

    if type == 0: # 处理计算结果
        equation = "\n".join(info['equation'])
        equation_answer = "\n".join(info['StepCalculatedCorrectlyResult'])
        if all(item == 1 for item in info['JudgmentStepCalculatedCorrectly']):
            # prompt = f"""我正在尝试检查一个数学问题的求解过程是否计算正确，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在检查的步骤是：{step}。\n\n 具体内容是{content}。\n\n 我识别到的公式有：{equation}。 \n\n 在不考虑计算公式是否正确的前提下，我认为计算结果是正确的。\n\n 请问我的判断是否正确？\n\n 如果判断正确你只需要回复<<yes>> \n\n 如果判断错误请做进一步的解释"""
            prompt = f"""I am trying to check that the solution process for a math problem is calculated correctly, the specific question is: {question} \n\n The steps I am currently using to solve the problem are as follows:{history}. \n\n The step I am currently checking is: {step}. \n\n The specific content is {content}. \n\n The formulas I recognized are:{equation}. \n\n Without considering whether the formula is correct or not, I think the calculation is correct. \n\n May I ask if my judgment is correct? \n\n If the judgment is correct you only need to reply <<yes>> \n\n If the judgment is wrong please explain further."""
        else:
            # prompt = f"""我正在尝试检查一个数学问题的求解过程是否计算正确，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在检查的步骤是：{step}。\n\n 具体内容是{content}。\n\n 我识别到的公式有：{equation}。 \n\n 在不考虑计算公式是否正确的前提下，我认为计算结果是错误的。\n\n 正确的结果应该是：{equation_answer} \n\n 请问我的判断是否正确？\n\n 如果判断正确你只需要回复<<yes>> \n\n 如果判断错误请做进一步的解释"""
            prompt = f"""I am trying to check that the solution process for a math problem is calculated correctly, the specific question is: {question} \n\n The steps I am currently using to solve the problem are as follows:{history}. \n\n The step I am currently checking is: {step}. \n\n The specific content is {content}. \n\n The formulas I recognized are:{equation}. \n\n Without considering whether the equation is correct or not, I think the result is wrong. \n\n The correct result should be: {equation_answer} \n\n Is my judgment correct? \n\n If the judgment is correct you only need to reply <<yes>> \n\n If the judgment is wrong please explain further"""

    elif type == 1: # 处理计算公式
        equation = "\n".join(info['equation'])
        correct_equation = "\n".join(info['StepEquationCorrectlyFormat'])
        if all(item == 1 for item in info['JudgmentStepEquationCorrectly']):
            # prompt = f"""我正在尝试检查一个数学问题的求解过程是否计算正确，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在检查的步骤是：{step}。\n\n 具体内容是{content}。\n\n 我识别到的公式有：{equation}。 \n\n 在不考虑计算结果是否正确的前提下，我认为计算公式是正确的。\n\n 请问我的判断是否正确？\n\n 如果判断正确你只需要回复<<yes>> \n\n 如果判断错误请做进一步的解释"""
            prompt = f"""I am trying to check that the solution process for a math problem is calculated correctly, the specific question is: {question} \n\n I am currently using the following steps to solve the problem: {history}. \n\n The step I am currently checking is: {step}. \n\n The specific content is {content}. \n\n The formulas I recognized are:{equation}. \n\n Without considering whether the result of the calculation is correct or not, I think the formula is correct. \n\n May I ask if my judgment is correct? \n\n If the judgment is correct you only need to reply <<yes>> \n\n If the judgment is wrong please explain further."""
        else:
            # prompt = f"""我正在尝试检查一个数学问题的求解过程是否计算正确，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在检查的步骤是：{step}。\n\n 具体内容是{content}。\n\n 我识别到的公式有：{equation}。 \n\n 在考虑计算结果是否正确之前，我认为计算公式本身就是是错误的。\n\n 正确的公式应该是：{correct_equation} \n\n 请问我的判断是否正确？\n\n 如果判断正确你只需要回复<<yes>> \n\n 如果判断错误请做进一步的解释"""
            prompt = f"""I am trying to check that the solution process for a math problem is calculated correctly, the specific question is: {question} \n\n The steps I am currently using to solve the problem are as follows:{history}. \n\n The step I am currently checking is: {step}. \n\n The specific content is {content}. \n\n The formulas that I have identified are:{equation}. \n\n Before considering whether the result of the calculation is correct, I assume that the formula itself is wrong. \n\n The correct equation would be: {correct_equation} \n\n Am I correct in my judgment? \n\n If the judgment is correct you only need to reply <<yes>> \n\n If the judgment is wrong please explain further."""
    
    elif type == 2: # 处理推理步骤
        if info['JudgmentStepReasoningCorrectly'] == 1:
            # prompt = f"""我正在尝试检查一个数学问题的求解过程是否推理合理。具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在检查的步骤是：{step}。\n\n 具体内容是{content}。\n\n 我认为这一步的推理是正确的。\n\n 请问我的判断是否正确？\n\n 如果判断正确你只需要回复<<yes>> \n\n 如果判断错误请做进一步的解释"""
            prompt = f"""I am trying to check if the process of solving a math problem is reasonably sound. The specific problem is: {question} \n\n I am currently using the following steps to solve the problem: {history}. \n\n The step I am checking right now is: {step}. \n\n The specifics are {content}. \n\n I think the reasoning in this step is correct. \n\n May I ask if my judgment is correct? \n\n If the judgment is correct you only need to reply <<yes>> \n\n If the judgment is wrong please explain further."""
        else:
            modify_content = info['StepReasoningCorrectlyResult']
            # prompt = f"""我正在尝试检查一个数学问题的求解过程是否推理合理。具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在检查的步骤是：{step}。\n\n 具体内容是{content}。\n\n 我认为这一步的推理是错误的，应该修改为：{modify_content} \n\n 请问我的判断是否正确？\n\n 如果判断正确你只需要回复<<yes>> \n\n 如果判断错误请做进一步的解释"""
            prompt = f"""I am trying to check if the process of solving a math problem is reasonably sound. The specific problem is: {question} \n\n The steps I am currently using to solve the problem are as follows:{history}. \n\n The step I am checking right now is: {step}. \n\n The specifics are {content}. \n\n I think the reasoning in this step is wrong and should be modified to read: {modify_content} \n\n Is my judgment correct? \n\n If the judgment is correct you only need to reply <<yes>> \n\n If the judgment is wrong please explain further"""

    for i in range(10):
        try:
            response = llm_response(prompt=prompt, backbone=backbone, url=url)
            return response
        except:
            response = ""
            
    return response

def analyze_data(json_data, processed_json_data, output_file_path, backbone = "chatglm_platform", url = CRITIC_URL):
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

        'LLMJudgmentStepCalculatedCorrectly': 0, # 统计计算步骤判断正确的步骤数（依据LLM判断）
        'LLMJudgmentStepEquationCorrectly': 0, # 统计计算公式判断正确的步骤数（依据LLM判断）
        'LLMJudgmentStepReasoningCorrectly': 0, # 统计推理步骤判断正确的步骤数（依据LLM判断）        

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
                if step_info.get('label', "") == 1:
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
                if type(response1) == str and len(response1) > 0:
                    response1_first_sentence = response1.split(".")[0]
                    if response1_first_sentence[:3].lower() == "yes" or "yes" in response1_first_sentence.lower():  # 如果计算正确
                        correct_judgments_by_step['LLMJudgmentStepCalculatedCorrectly'] += 1
                    else:
                        llm_total_calculate_correct = False
                        explain_correct = False


                # 检测LLM针对计算公式的判断是否正确
                response1 = step_info['LLMJudgmentStepEquationCorrectly']
                # 获取response的第一句话或者如果没有符号就是完整的response
                if type(response1) == str and len(response1) > 0:
                    response1_first_sentence = response1.split(".")[0]
                    if response1_first_sentence[:3].lower() == "yes" or "yes" in response1_first_sentence.lower():  # 如果计算正确
                        correct_judgments_by_step['LLMJudgmentStepEquationCorrectly'] += 1
                    else:
                        llm_total_equation_correct = False
                        explain_correct = False

            else:

                correct_judgments_by_step['reasoning_steps'] += 1
                has_reasoning = True

                if step_info.get('label', "") == 1:
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
                if type(response2) == str and len(response2) > 0:
                    response2_first_sentence = response2.split(".")[0]
                    if response2_first_sentence[:3].lower() == "yes" or "yes" in response2_first_sentence.lower():  # 如果推理正确
                        correct_judgments_by_step['LLMJudgmentStepReasoningCorrectly'] += 1
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
            correct_judgments_by_case['LabelJudgmentCaseCalculatedCorrectly'] += 1
        if label_total_equation_correct == True:
            correct_judgments_by_case['LabelJudgmentCaseEquationCorrectly'] += 1
        if label_total_reasoning_correct == True:
            correct_judgments_by_case['LabelJudgmentCaseReasoningCorrectly'] += 1

        if llm_total_calculate_correct == True:
            correct_judgments_by_case['LLMJudgmentCaseCalculatedCorrectly'] += 1
        if llm_total_equation_correct == True:
            correct_judgments_by_case['LLMJudgmentCaseEquationCorrectly'] += 1
        if llm_total_reasoning_correct == True:
            correct_judgments_by_case['LLMJudgmentCaseReasoningCorrectly'] += 1
        
        if total_python_used == True:
            correct_judgments_by_case['python_code_count'] += 1
        if total_sympy_used == True:
            correct_judgments_by_case['sympy_count'] += 1



    # for entry in json_data:
    for entry in tqdm(json_data, desc='Processing'):
        if entry['id'] in processed_ids:
            continue  # 跳过已处理的数据
        
        correct_judgments_by_case['total_cases'] += 1
        question = entry['question']
        
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

            step_info['LLMJudgmentStepEquationCorrectly'] = ""
            step_info['LLMJudgmentStepCalculatedCorrectly'] = ""
            step_info['LLMJudgmentStepReasoningCorrectly'] = ""

            # 计算总步骤数
            correct_judgments_by_step['total_steps'] += 1
            
            # 检查每种判断类型
            if step_info['is_calculation_or_reasoning'] == 1:
                correct_judgments_by_step['calculation_steps'] += 1
                has_calculation = True
                Flag = False
                # 检测计算步的标签是否正确
                if step_info.get('label', "") == 1:
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
                step_info['LLMJudgmentStepCalculatedCorrectly'] = llm_judge_response(question, step_key, step_info, 0, backbone, url)  # 调用生成方法
                response1 = step_info['LLMJudgmentStepCalculatedCorrectly']
                # 获取response的第一句话或者如果没有符号就是完整的response
                if type(response1) == str and len(response1) > 0:
                    response1_first_sentence = response1.split(".")[0]
                    if response1_first_sentence[:3].lower() == "yes" or "yes" in response1_first_sentence.lower():  # 如果计算正确
                        correct_judgments_by_step['LLMJudgmentStepCalculatedCorrectly'] += 1
                    else:
                        llm_total_calculate_correct = False
                        explain_correct = False


                # 检测LLM针对计算公式的判断是否正确
                step_info['LLMJudgmentStepEquationCorrectly'] = llm_judge_response(question, step_key, step_info, 1, backbone, url)  # 调用生成方法  
                response1 = step_info['LLMJudgmentStepEquationCorrectly']
                # 获取response的第一句话或者如果没有符号就是完整的response
                if type(response1) == str and len(response1) > 0:
                    response1_first_sentence = response1.split(".")[0]
                    if response1_first_sentence[:3].lower() == "yes" or "yes" in response1_first_sentence.lower():  # 如果计算正确
                        correct_judgments_by_step['LLMJudgmentStepEquationCorrectly'] += 1
                    else:
                        llm_total_equation_correct = False
                        explain_correct = False

            else:

                correct_judgments_by_step['reasoning_steps'] += 1
                has_reasoning = True

                if step_info.get('label', "") == 1:
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
                step_info['LLMJudgmentStepReasoningCorrectly'] = llm_judge_response(question, step_key, step_info, 2, backbone, url)  # 调用
                response2 = step_info['LLMJudgmentStepReasoningCorrectly']
                # 获取response的第一句话或者如果没有符号就是完整的response
                if type(response2) == str and len(response2) > 0:   
                    response2_first_sentence = response2.split(".")[0]
                    if response2_first_sentence[:3].lower() == "yes" or "yes" in response2_first_sentence.lower():  # 如果推理正确
                        correct_judgments_by_step['LLMJudgmentStepReasoningCorrectly'] += 1
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
            correct_judgments_by_case['LabelJudgmentCaseCalculatedCorrectly'] += 1
        if label_total_equation_correct == True:
            correct_judgments_by_case['LabelJudgmentCaseEquationCorrectly'] += 1
        if label_total_reasoning_correct == True:
            correct_judgments_by_case['LabelJudgmentCaseReasoningCorrectly'] += 1

        if llm_total_calculate_correct == True:
            correct_judgments_by_case['LLMJudgmentCaseCalculatedCorrectly'] += 1
        if llm_total_equation_correct == True:
            correct_judgments_by_case['LLMJudgmentCaseEquationCorrectly'] += 1
        if llm_total_reasoning_correct == True:
            correct_judgments_by_case['LLMJudgmentCaseReasoningCorrectly'] += 1
        
        if total_python_used == True:
            correct_judgments_by_case['python_code_count'] += 1
        if total_sympy_used == True:
            correct_judgments_by_case['sympy_count'] += 1
        
        # 处理完毕后，将数据追加到新文件
        append_jsonl(entry, output_file_path)
        
    return correct_judgments_by_step, correct_judgments_by_case

def print_padded_line(key, value, explanation, width=60):
    """格式化输出一行，并附带解释，确保总宽度为固定值"""
    line = f"{key}: {value}"
    print(line + ' ' * (width - len(line)) + explanation)

def print_statistics(stats_by_step, stats_by_case, output_file_path):
    """打印统计信息为表格形式，并格式化为固定宽度的列，使用print函数，并考虑字符宽度，同时附带每项数据的解释。"""

    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['键', '值', '说明'])

        print("步骤统计数据:")
        writer.writerow(["步骤统计数据"])
        explanation = [
            "总计算步数",
            "总推理步数",
            "规则判断计算正确的步数",
            "规则判断公式正确的步数",
            "规则判断推理正确的步数",
            "LLM判断计算正确的步数",
            "LLM判断公式正确的步数",
            "LLM判断推理正确的步数",
            "正确标识计算标签的步数",
            "正确表示推理标签的步数",
            "总步数",
            "使用SymPy的步数",
            "使用Python编程的步数",
        ]
        for i, (key, value) in enumerate(stats_by_step.items()):
            print_padded_line(key, value, explanation[i])
            writer.writerow([key, value, explanation[i]])
        print('-' * 60)  # 根据总宽度调整分隔线长度

        print("案例统计数据:")
        writer.writerow(["案例统计数据"])
        explanation = [
            "只有计算步骤的案例数",
            "既有计算步骤又有推理步骤的案例数",
            "只有推理步骤的案例数",
            "规则判断计算正确的案例数",
            "规则判断公式正确的案例数",
            "规则判断推理正确的案例数",
            "LLM判断计算正确的案例数",
            "LLM判断公式正确的案例数",
            "LLM判断推理正确的案例数",
            "正确标识计算标签的案例数",
            "正确表示推理标签的案例数",
            "总案例数",
            "使用SymPy的案例数",
            "使用Python编程的案例数",
        ]
        for i, (key, value) in enumerate(stats_by_case.items()):
            print_padded_line(key, value, explanation[i])
            writer.writerow([key, value, explanation[i]])
        print('-' * 60)  # 根据总宽度调整分隔线长度

        # 用于存储正确判断的步骤数
        total_steps = stats_by_step.get('total_steps', 0)
        if total_steps > 0:
            print("步骤统计比率及其解释:")
            writer.writerow(["步骤统计比率及其解释"])
            for key, value in stats_by_step.items():
                if key != 'total_steps':  # 避免除以自己
                    explanation = {
                        'calculation_steps': "完成的计算步骤总数占比",
                        'reasoning_steps': "完成的推理步骤总数占比",
                        'LabelJudgmentStepCalculatedCorrectly': "根据人工标签正确计算的步骤占比",
                        'LabelJudgmentStepEquationCorrectly': "根据人工标签正确的计算公式步骤占比",
                        'LabelJudgmentStepReasoningCorrectly': "根据人工标签正确的推理步骤占比",
                        'LLMJudgmentStepCalculatedCorrectly': "LLM判断计算步骤正确的占比",
                        'LLMJudgmentStepEquationCorrectly': "LLM判断计算公式正确的步骤占比",
                        'LLMJudgmentStepReasoningCorrectly': "LLM判断推理步骤正确的占比",
                        'correct_label_steps': "标签正确的总步骤占比",
                        'correct_explain_steps': "解释正确的总步骤占比",
                        'sympy_count': "使用SymPy的步骤占比",
                        'python_code_count': "使用Python编程的步骤占比",
                    }.get(key, "未知统计指标")
                    percent_value = value / total_steps * 100
                    print_padded_line(f"{key} 步骤占比", f"{percent_value:.2f}%", explanation)
                    writer.writerow([f"{key} 步骤占比", f"{percent_value:.2f}%", explanation])
            print("细分步骤统计比率及其解释:")
            writer.writerow(["细分步骤统计比率及其解释"])
            # 处理分母可能为0的情况
            if stats_by_step["calculation_steps"] > 0:
                temp = min(stats_by_step["LabelJudgmentStepCalculatedCorrectly"] / stats_by_step["calculation_steps"] * 100, 100)
                print_padded_line(f"LabelJudgmentStepCalculatedCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确计算的计算步骤占比")
                writer.writerow([f"LabelJudgmentStepCalculatedCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确计算的计算步骤占比"])
            else:
                print_padded_line(f"LabelJudgmentStepCalculatedCorrectly 步骤占比", "N/A", "根据人工标签正确计算的计算步骤占比")
                writer.writerow([f"LabelJudgmentStepCalculatedCorrectly 步骤占比", "N/A", "根据人工标签正确计算的计算步骤占比"])
            if stats_by_step["calculation_steps"] > 0:
                temp = min(stats_by_step["LabelJudgmentStepEquationCorrectly"] / stats_by_step["calculation_steps"] * 100, 100)
                print_padded_line(f"LabelJudgmentStepEquationCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确的计算公式占比")
                writer.writerow([f"LabelJudgmentStepEquationCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确的计算公式占比"])
            else:
                print_padded_line(f"LabelJudgmentStepEquationCorrectly 步骤占比", "N/A", "根据人工标签正确的计算公式占比")
                writer.writerow([f"LabelJudgmentStepEquationCorrectly 步骤占比", "N/A", "根据人工标签正确的计算公式占比"])
            if stats_by_step["reasoning_steps"] > 0:
                temp = min(stats_by_step["LabelJudgmentStepReasoningCorrectly"] / stats_by_step["reasoning_steps"] * 100, 100)
                print_padded_line(f"LabelJudgmentStepReasoningCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确推理的推理步骤占比")
                writer.writerow([f"LabelJudgmentStepReasoningCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确推理的推理步骤占比"])
            else:
                print_padded_line(f"LabelJudgmentStepReasoningCorrectly 步骤占比", "N/A", "根据人工标签正确推理的推理步骤占比")
                writer.writerow([f"LabelJudgmentStepReasoningCorrectly 步骤占比", "N/A", "根据人工标签正确推理的推理步骤占比"])
            if stats_by_step["calculation_steps"] > 0:
                temp = min(stats_by_step["LLMJudgmentStepCalculatedCorrectly"] / stats_by_step["calculation_steps"] * 100, 100)
                print_padded_line(f"LLMJudgmentStepCalculatedCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确计算的计算步骤占比")
                writer.writerow([f"LLMJudgmentStepCalculatedCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确计算的计算步骤占比"])
            else:
                print_padded_line(f"LLMJudgmentStepCalculatedCorrectly 步骤占比", "N/A", "LLM判断正确计算的计算步骤占比")
                writer.writerow([f"LLMJudgmentStepCalculatedCorrectly 步骤占比", "N/A", "LLM判断正确计算的计算步骤占比"])
            if stats_by_step["calculation_steps"] > 0:
                temp = min(stats_by_step["LLMJudgmentStepEquationCorrectly"] / stats_by_step["calculation_steps"] * 100, 100)
                print_padded_line(f"LLMJudgmentStepEquationCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确公式占比")
                writer.writerow([f"LLMJudgmentStepEquationCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确公式占比"])
            else:
                print_padded_line(f"LLMJudgmentStepEquationCorrectly 步骤占比", "N/A", "LLM判断正确公式占比")
                writer.writerow([f"LLMJudgmentStepEquationCorrectly 步骤占比", "N/A", "LLM判断正确公式占比"])
            if stats_by_step["reasoning_steps"] > 0:
                temp = min(stats_by_step["LLMJudgmentStepReasoningCorrectly"] / stats_by_step["reasoning_steps"] * 100, 100)
                print_padded_line(f"LLMJudgmentStepReasoningCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确推理步骤占比")
                writer.writerow([f"LLMJudgmentStepReasoningCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确推理步骤占比"])
            else:
                print_padded_line(f"LLMJudgmentStepReasoningCorrectly 步骤占比", "N/A", "LLM判断正确推理步骤占比")
                writer.writerow([f"LLMJudgmentStepReasoningCorrectly 步骤占比", "N/A", "LLM判断正确推理步骤占比"])

        # 用于存储正确判断的样例数
        total_cases = stats_by_case['total_cases']
        if total_cases > 0:
            print("案例统计比率及其解释:")
            writer.writerow(["案例统计比率及其解释"])
            for key, value in stats_by_case.items():
                if key != 'total_cases':  # 避免除以自己
                    explanation = {
                        'only_calculation_cases': "仅包含计算步骤的样例占比",
                        'calculation_and_reasoning_cases': "包含计算和推理步骤的样例占比",
                        'only_reasoning_cases': "仅包含推理步骤的样例占比",
                        'LabelJudgmentCaseCalculatedCorrectly': "人工标签判断计算正确的样例占比",
                        'LabelJudgmentCaseEquationCorrectly': "人工标签判断计算公式正确的样例占比",
                        'LabelJudgmentCaseReasoningCorrectly': "人工标签判断推理正确的样例占比",
                        'LLMJudgmentCaseCalculatedCorrectly': "LLM判断计算步骤正确的样例占比",
                        'LLMJudgmentCaseEquationCorrectly': "LLM判断计算公式正确的样例占比",
                        'LLMJudgmentCaseReasoningCorrectly': "LLM判断推理步骤正确的样例占比",
                        'correct_label_cases': "标签判断整体正确的样例占比",
                        'correct_explain_cases': "解释整体正确的样例占比",
                        'sympy_count': "使用SymPy的样例占比",
                        'python_code_count': "使用Python编程的样例占比",
                    }.get(key, "未知统计指标")
                    percent_value = value / total_cases * 100
                    print_padded_line(f"{key} 样例占比", f"{percent_value:.2f}%", explanation)
                    writer.writerow([f"{key} 样例占比", f"{percent_value:.2f}%", explanation])
            print("细分样例统计比率及其解释:")
            writer.writerow(["细分样例统计比率及其解释"])
            if (stats_by_case["only_calculation_cases"]  + stats_by_case["calculation_and_reasoning_cases"]) > 0:
                temp = min(stats_by_case["LabelJudgmentCaseCalculatedCorrectly"] / (stats_by_case["only_calculation_cases"]  + stats_by_case["calculation_and_reasoning_cases"])  * 100, 100)
                print_padded_line(f"LabelJudgmentCaseCalculatedCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确计算的计算样例占比")
                writer.writerow([f"LabelJudgmentCaseCalculatedCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确计算的计算样例占比"])
                temp = min(stats_by_case["LabelJudgmentCaseEquationCorrectly"] / (stats_by_case ["only_calculation_cases"] + stats_by_case["calculation_and_reasoning_cases"]) * 100, 100)
                print_padded_line(f"LabelJudgmentCaseEquationCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确的计算公式样例占比")
                writer.writerow([f"LabelJudgmentCaseEquationCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确的计算公式样例占比"])
            else:
                print_padded_line(f"LabelJudgmentCaseCalculatedCorrectly 步骤占比", "N/A", "根据人工标签正确计算的计算样例占比")
                writer.writerow([f"LabelJudgmentCaseCalculatedCorrectly 步骤占比", "N/A", "根据人工标签正确计算的计算样例占比"])
                print_padded_line(f"LabelJudgmentCaseEquationCorrectly 步骤占比", "N/A", "根据人工标签正确的计算公式样例占比")
                writer.writerow([f"LabelJudgmentCaseEquationCorrectly 步骤占比", "N/A", "根据人工标签正确的计算公式样例占比"])
            if (stats_by_case["only_reasoning_cases"] + stats_by_case["calculation_and_reasoning_cases"]) > 0:
                temp = min(stats_by_case["LabelJudgmentCaseReasoningCorrectly"] / (stats_by_case["only_reasoning_cases"] + stats_by_case["calculation_and_reasoning_cases"]) * 100, 100)
                print_padded_line(f"LabelJudgmentCaseReasoningCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确推理的样例占比")
                writer.writerow([f"LabelJudgmentCaseReasoningCorrectly 步骤占比", f"{temp:.2f}%", "根据人工标签正确推理的样例占比"])
            else:
                print_padded_line(f"LabelJudgmentCaseReasoningCorrectly 步骤占比", "N/A", "根据人工标签正确推理的样例占比")
                writer.writerow([f"LabelJudgmentCaseReasoningCorrectly 步骤占比", "N/A", "根据人工标签正确推理的样例占比"])
            if (stats_by_case["only_calculation_cases"]  + stats_by_case["calculation_and_reasoning_cases"]) > 0:
                temp = min(stats_by_case["LLMJudgmentCaseCalculatedCorrectly"] / (stats_by_case["only_calculation_cases"]  + stats_by_case["calculation_and_reasoning_cases"]) * 100, 100)
                print_padded_line(f"LLMJudgmentCaseCalculatedCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确计算的样例步骤占比")
                writer.writerow([f"LLMJudgmentCaseCalculatedCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确计算的样例步骤占比"])
                temp = min(stats_by_case["LLMJudgmentCaseEquationCorrectly"] / (stats_by_case["only_calculation_cases"]  + stats_by_case["calculation_and_reasoning_cases"]) * 100, 100)
                print_padded_line(f"LLMJudgmentCaseEquationCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确公式的样例占比")
                writer.writerow([f"LLMJudgmentCaseEquationCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确公式的样例占比"])
            else:
                print_padded_line(f"LLMJudgmentCaseCalculatedCorrectly 步骤占比", "N/A", "LLM判断正确计算的样例步骤占比")
                writer.writerow([f"LLMJudgmentCaseCalculatedCorrectly 步骤占比", "N/A", "LLM判断正确计算的样例步骤占比"])
                print_padded_line(f"LLMJudgmentCaseEquationCorrectly 步骤占比", "N/A", "LLM判断正确公式的样例占比")
                writer.writerow([f"LLMJudgmentCaseEquationCorrectly 步骤占比", "N/A", "LLM判断正确公式的样例占比"])
            if (stats_by_case["only_reasoning_cases"] + stats_by_case["calculation_and_reasoning_cases"]) > 0:
                temp = min(stats_by_case["LLMJudgmentCaseReasoningCorrectly"] / (stats_by_case["only_reasoning_cases"] + stats_by_case["calculation_and_reasoning_cases"]) * 100, 100)
                print_padded_line(f"LLMJudgmentCaseReasoningCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确推理的样例占比")
                writer.writerow([f"LLMJudgmentCaseReasoningCorrectly 步骤占比", f"{temp:.2f}%", "LLM判断正确推理的样例占比"])
            else:
                print_padded_line(f"LLMJudgmentCaseReasoningCorrectly 步骤占比", "N/A", "LLM判断正确推理的样例占比")
                writer.writerow([f"LLMJudgmentCaseReasoningCorrectly 步骤占比", "N/A", "LLM判断正确推理的样例占比"])

def Check2_CalculateAccuracy(input_file_path, backbone = "chatglm_platform", url = CRITIC_URL):
    # 根据需要修改文件路径

    # 检查filename中"Step"的位置并插入"Check2"
    output_file_path = input_file_path.replace("Step4", "Check2Step4").replace("step4", "Check2Step4")
    
    processed_data = read_processed_jsonl(output_file_path)
    json_data = read_jsonl(input_file_path)
    correct_judgments_by_step, correct_judgments_by_case = analyze_data(json_data, processed_data, output_file_path, backbone, url)
    
    # 打印统计信息
    output_file_path2 = output_file_path.replace(".jsonl", "_statistics.csv")
    print_statistics(correct_judgments_by_step, correct_judgments_by_case, output_file_path2)

def main():
    # input_file_path  = 'F://code//github//ChatGLM-MathV2//data//peiyi9979_Math_Shepherd_for_codeTest_Step4_JudgmentStepReasoningCorrectly//math-shepherd.jsonl_1-10.jsonl'
    input_file_path = "F://code//github//ChatGLM-MathV2//data//test_data100//front_step4//test_data100.jsonl"
    Check2_CalculateAccuracy(input_file_path)

if __name__ == "__main__":
    main()
