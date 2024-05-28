# 如果打开下面一行，命令行会自动输出代码执行的时间
import time
# 这个装饰器 time_it 可以被应用到任何你希望测量执行时间的函数上。它通过计算函数开始和结束时的时间来计算执行时间，并将时间转换为小时、分钟和秒的格式。
def time_it(func):
    """
    装饰器，用于测量函数执行时间。
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 获取开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 获取结束时间
        time_taken = end_time - start_time  # 计算耗时
        hours, rem = divmod(time_taken, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"{func.__name__} executed in: {int(hours):02d}h:{int(minutes):02d}m:{seconds:06.3f}s")
        return result
    return wrapper


import os
import json
import re
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm

from utils.get_data_for_codeTest import get_data_for_codeTest
from Step1_SplitByRow_forMathShepherd import Step1_SplitByRow_forMathShepherd
from Step2_IsCalculationOrReasoning import Step2_IsCalculationOrReasoning
from Step3_JudgmentStepCalculatedCorrectly import Step3_JudgmentStepCalculatedCorrectly, llm_response, TGI_URL, CRITIC_URL
from Check1_JsonlVisualization import Check1_JsonlVisualization
from Check2_CalculateAccuracy import Check2_CalculateAccuracy


# 判断单步推理是否正确
def check_reasoning(per_step, content, question, history, backbone = "tgi", url = TGI_URL):
    # promt = f"""我正在尝试解决一个数学问题，具体问题是：“{question}”。\n\n 我之前的解题步骤如下：“{history}” \n\n 现在我正在推理这一步是：“{per_step}”，具体推理内容是：“{content}”。\n\n 请评估我这一步的推理是否正确。\n\n 如果我目前的推理步骤正确并且与问题相关，只需要回答“yes”。\n\n 如果这一步推理错误或者不相关，需要你进行修正，并直接提供正确或更相关的推理步骤，用<<>>包围起来。"""
    prompt = f"""I am trying to solve a math problem, the specific question is: "{question}". \n\n My previous step in solving the problem was as follows:"{history}" \n\n Now I'm reasoning that this step is:"{per_step}" and the specific reasoning is:"{content}". \n\n Please evaluate if my reasoning in this step is correct. \n\n If my current reasoning step is correct and relevant to the question, just answer "yes". \n\n If this step of reasoning is incorrect or irrelevant, you are required to correct it and provide the correct or more relevant step of reasoning directly, surrounded by <<>>."""
    for i in range(10):
        # response = gpt_generate(prompt)
        response = llm_response(prompt=prompt, backbone=backbone, url=url)  # 调用生成方法
        if type(response) == str and len(response) > 0:
            # 获取response的第一句话或者如果没有符号就是完整的response
            response_first_sentence = response.split(".")[0]
            # 提取 response 的前三个字符，并将它们转换成小写来进行判断。
            if response_first_sentence[:3].lower() == "yes" or "yes" in response_first_sentence.lower():  
                return 1, response
            else:
                # 尝试修正表达式
                match = re.search(r'<<(.+?)>>', response)
                if match:
                    return 0, match.group(1)
                else:
                    continue
        else:
            continue
    return 0, ""

# 串行处理
def process_jsonl_file(source_path, dest_path):
    with open(source_path, 'r', encoding='utf-8') as src_file, \
         open(dest_path, 'w', encoding='utf-8') as dest_file:
        for line in tqdm(src_file, desc='Processing'):
            data = json.loads(line)
            # 遍历每一步的解决方案
            if 'solution' in data:
                # 获取历史信息
                history_json = data['history_json']
                history = ""
                question = data['question']
                for step, info in data['solution'].items(): # step变量会接收步骤的名称（如"Step 1"），而info变量会接收与这个步骤名称对应的字典值。
                    if info['is_calculation_or_reasoning'] == 1: # 如果是计算步
                        info['JudgmentStepReasoningCorrectly'], info['StepReasoningCorrectlyResult'] = 1, "This is a calculation step."
                    else:
                        # 判断并添加新键
                        info['JudgmentStepReasoningCorrectly'], info['StepReasoningCorrectlyResult'] = check_reasoning(step, info['content'], question, history)
                        # 添加到历史信息中
                        if info['JudgmentStepReasoningCorrectly'] == 1:
                            history_json[step] = info['content']
                        else:
                            history_json[step] = info['StepReasoningCorrectlyResult']
                    history += f"{step}: {history_json[step]}\n"

            # 将修改后的数据写回新的JSONL文件
            json.dump(data, dest_file, ensure_ascii=False)
            dest_file.write('\n')

# 并发处理
def process_line(line, backbone = "tgi", url = TGI_URL):
    data = json.loads(line)
    if 'solution' in data:
        # 获取历史信息
        history = ""
        question = data['question']
        # 根据全部步骤判断整个题目的正确性
        data['JudgmentAllCorrectly'] = 1
        for step, info in data['solution'].items(): # step变量会接收步骤的名称（如"Step 1"），而info变量会接收与这个步骤名称对应的字典值。
            history_json = info['history_json']
            if info['is_calculation_or_reasoning'] == 1: # 如果是计算步
                info['JudgmentStepReasoningCorrectly'], info['StepReasoningCorrectlyResult'] = 1, "This is a calculation step."
                # 判断整个题目的正确性
                if any(item for item in info['JudgmentStepCalculatedCorrectly']) == 0:
                    data['JudgmentAllCorrectly'] = 0
                elif any(item for item in info['JudgmentStepEquationCorrectly']) == 0:
                    data['JudgmentAllCorrectly'] = 0
            else:
                # 判断并添加新键
                info['JudgmentStepReasoningCorrectly'], info['StepReasoningCorrectlyResult'] = check_reasoning(step, info['content'], question, history, backbone, url)
                # 添加到历史信息中
                if info['JudgmentStepReasoningCorrectly'] == 1:
                    history_json[step] = info['content']
                else:
                    history_json[step] = info['StepReasoningCorrectlyResult']
                # 判断整个题目的正确性
                if info['JudgmentStepReasoningCorrectly'] == 0:
                    data['JudgmentAllCorrectly'] = 0
            history += f"{step}: {history_json[step]}\n"

    # 将处理后的数据转换为字符串以便写入文件
    return json.dumps(data, ensure_ascii=False) + '\n'

def process_jsonl_file_concurrent(source_path, dest_path):
    # 读取文件的所有行
    with open(source_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    results = []

    # 使用 ThreadPoolExecutor 来并发处理每一行
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有行到线程池
        future_to_line = {executor.submit(process_line, line): line for line in tqdm(lines, desc='Processing')}

        # 使用tqdm创建进度条
        with tqdm(total=len(future_to_line), desc='Processing lines') as progress:
            # 收集处理结果
            for future in concurrent.futures.as_completed(future_to_line):
                results.append(future.result())
                progress.update(1)  # 更新进度条
    
     # 写入结果到目标文件
    with open(dest_path, 'w', encoding='utf-8') as file:
        file.writelines(results)

# 实现每1000条数据保存一次，并且能够在之后的运行中从上次结束的地方继续开始处理
def process_jsonl_file_concurrent2(source_path, dest_path, chunk_size=1000, start_from=0):
    # 读取文件的所有行
    with open(source_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 只处理从 start_from 开始的数据
    lines = lines[start_from:]

    results = []

   # 创建线程池进行并行处理
    with ThreadPoolExecutor(max_workers=50) as executor:
        # 从 start_from 开始批量处理数据
        for chunk_start in range(0, len(lines), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(lines))
            future_to_line = {executor.submit(process_line, line): line for line in lines[chunk_start:chunk_end]}
            results = []
            
            # 使用 tqdm 创建进度条
            with tqdm(total=len(future_to_line), desc=f'Processing lines {chunk_start+1} to {chunk_end}') as progress:
                for future in as_completed(future_to_line):
                    results.append(future.result())
                    progress.update(1)  # 更新进度条
            
            # 保存这一批数据到文件
            save_file_name = f"{dest_path}_{chunk_start+1}-{chunk_end}.jsonl"
            with open(save_file_name, 'w', encoding='utf-8') as f:
                    f.writelines(results)
            print(f"Data saved to {save_file_name}")
            
            # 可视化结果输出，用于debug
            Check1_JsonlVisualization(save_file_name)

            # 计算acc
            Check2_CalculateAccuracy(save_file_name)

# 采用set来区分已处理和未处理的行
def process_jsonl_file_concurrent3(source_path, dest_path, max_workers=10, backbone = "tgi", url = TGI_URL):
    processed_questions = set()
    
    # 读取已处理的问题
    if os.path.exists(dest_path):
        with open(dest_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    processed_questions.add(data['question'])
                except json.JSONDecodeError:
                    continue

    # 读取源文件并过滤已处理的行
    lines_to_process = []
    with open(source_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                if data['question'] not in processed_questions:
                    lines_to_process.append(line)
            except json.JSONDecodeError:
                continue
    
    # 使用 ThreadPoolExecutor 来并发处理每一行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_line, line, backbone, url): line for line in lines_to_process}
        
        with tqdm(total=len(futures), desc='Processing lines') as progress:
            with open(dest_path, 'a', encoding='utf-8') as output_file:
                for future in as_completed(futures):
                    result = future.result()
                    output_file.write(result)
                    output_file.flush()  # 确保每次写入后立即保存
                    progress.update(1)  # 更新进度条



@time_it
def Step4_JudgmentStepReasoningCorrectly(source_folder, target_folder, max_workers = 10, backbone1 = "tgi", url1 = TGI_URL, backbone2 = "chatglm_platform", url2 = CRITIC_URL):
    
    print("Step4: 判断步骤是否推理正确……")

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jsonl'):
            source_path = os.path.join(source_folder, filename)
            print("正在处理文件:", source_path)
            dest_path = os.path.join(target_folder, filename)
            # process_jsonl_file(source_path, dest_path)
            # process_jsonl_file_concurrent(source_path, dest_path)
            process_jsonl_file_concurrent3(source_path, dest_path, max_workers, backbone1, url1)

            # 可视化结果输出，用于debug
            Check1_JsonlVisualization(dest_path)

            # 计算acc
            Check2_CalculateAccuracy(dest_path, backbone2, url2)

# 使用方法：
def main2():
    code_test_state = True
    base_folder = "F://code//github//ChatGLM-MathV2"
    # base_folder = "/workspace/dujh22/ChatGLM-MathV2"
    dataset_name = "peiyi9979_Math_Shepherd"
    source_folder = base_folder + '//raw_data//' + dataset_name

    mid_name = base_folder + '//data//' + dataset_name

    if code_test_state:
        #get_data_for_codeTest(source_folder, new_folder_suffix='_for_codeTest', num_points=10)
        source_folder = source_folder + "_for_codeTest"

        target_folder1 = mid_name + "_for_codeTest" + "_Step1_SplitByRow_forMathShepherd"
        target_folder2 = mid_name + "_for_codeTest" + "_Step2_IsCalculationOrReasoning"
        target_folder3 = mid_name + "_for_codeTest" + "_Step3_JudgmentStepCalculatedCorrectly"
        target_folder4 = mid_name + "_for_codeTest" + "_Step4_JudgmentStepReasoningCorrectly"
    else:
        target_folder1 = mid_name + "_Step1_SplitByRow_forMathShepherd"
        target_folder2 = mid_name + "_Step2_IsCalculationOrReasoning"
        target_folder3 = mid_name + "_Step3_JudgmentStepCalculatedCorrectly"
        target_folder4 = mid_name + "_Step4_JudgmentStepReasoningCorrectly"

    #Step1_SplitByRow_forMathShepherd(source_folder, target_folder1)
    #Step2_IsCalculationOrReasoning(target_folder1, target_folder2)
    #Step3_JudgmentStepCalculatedCorrectly(target_folder2, target_folder3)
    Step4_JudgmentStepReasoningCorrectly(target_folder3, target_folder4)

def main():
    if len(sys.argv) > 7:
        source_folder = sys.argv[1]
        target_folder = sys.argv[2]
        max_workers = int(sys.argv[3])
        backbone1 = sys.argv[4]
        url1 = sys.argv[5]
        backbone2 = sys.argv[6]
        url2 = sys.argv[7]
    else:
        backbone1 = "tgi"
        url1 = TGI_URL
        backbone2 = "chatglm_platform"
        url2 = CRITIC_URL
        source_folder = 'F://code//github//ChatGLM-MathV2//data//test_data100//front_step3'
        target_folder = 'F://code//github//ChatGLM-MathV2//data//test_data100//front_step4' 
        max_workers = 10
    Step4_JudgmentStepReasoningCorrectly(source_folder, target_folder, max_workers, backbone1, url1, backbone2, url2)

if __name__ == '__main__':
    main()