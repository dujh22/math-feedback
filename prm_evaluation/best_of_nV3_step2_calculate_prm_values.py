import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime

from llm.llm_response import llm_response, TGI_URL, CRITIC_URL
from shepherd_prm.query_api import critic_math_problem, prepare_template

from prm_evaluation.openai.simple_evals.math_eval import QUERY_TEMPLATE, ANSWER_PATTERN, EQUALITY_TEMPLATE

PROMPT_TEMPLATE = prepare_template('/workspace/dujh22/math_feedback/shepherd_prm/templates/criticllm_math_template.txt')

def get_best_of_n_reward_prod(question, response, answer, backbone, url, id, idx):
    # 2.llm_answer
    # get the answer
    # temp_response = response.replace("$ANSWER (without quotes) where $ANSWER is the answer to the problem.", "")
    # match = re.search(ANSWER_PATTERN, temp_response)
    pattern = re.compile(r"The answer is:\s*(.*?)\s*ки")
    match = pattern.search(response)
    
    extracted_answer = match.group(1) if match else None
    prompt = EQUALITY_TEMPLATE % {"expression1": extracted_answer, "expression2": answer}
    llm_answer_flag = 0
    print(f"2.get {id} - {idx} response answer judgements-----------")
    for i in range(3):
        temp_response = llm_response(prompt, backbone, url)
        if temp_response.lower().strip() == "yes":
            llm_answer_flag = 1
            break

    # 3.llm_response
    print(f"3.get {id} - {idx} response judgements-----------")
    item = {"question": question, "response": response, "answer": answer}
    item = critic_math_problem(x=item, backbone=backbone, prompt_key="question", response_key="response", reference_key="answer", max_retry=3, PROMPT_TEMPLATE=PROMPT_TEMPLATE,  url=url) # get the critic result
    llm_response_flag = 0
    if item.get("critic_result") is not None:
        grade = float(item["critic_result"][0]["rating"])
        if grade >= 9:
           llm_response_flag = 1

    return idx, extracted_answer, llm_answer_flag, llm_response_flag

def load_data(input_file_path, data_length):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp_data = json.loads(line.strip())
            data.append(temp_data)
            if len(data) >= data_length:
                break
    return data

def calculate_prm_values(data, critic_backbone, critic_url, max_workers_num, output_file_path):
    # # 假设已经跑过一次拥有了结果，只是希望合并进来
    # raw_data = []
    # with open("/workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/test_rm3.jsonl", 'r', encoding='utf-8') as f0:
    #     for line in f0:
    #         temp_data = json.loads(line.strip())
    #         raw_data.append(temp_data)
    # with open(output_file_path, 'w', encoding='utf-8') as f:
    #     for item in tqdm(data, desc='Calculating N PRMvalue'):
    #         reference_item = next((i for i in raw_data if i.get('unique_id') == item.get('unique_id')), None)
    #         item['extracted_answer'] = reference_item["extracted_answer"]
    #         item['llm_answer_flag'] = reference_item['llm_answer_flag']
    #         item['llm_response_flag'] = reference_item['llm_response_flag']
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    #         f.flush()
    # return data
    # 否则，直接跑
    id = 0
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc='Calculating N PRMvalue'):
            id += 1
            if item.get('extraced_answer') is not None:
                continue
            item['extracted_answer'] = []
            item['llm_answer_flag'] = []
            item['llm_response_flag'] = []
            temp_answer = {}
            with ThreadPoolExecutor(max_workers=max_workers_num) as executor:
                futures = {
                    executor.submit(
                        get_best_of_n_reward_prod, 
                        item['question'], 
                        response, 
                        item['answer'], 
                        critic_backbone, 
                        critic_url, 
                        id, 
                        idx
                    ): response 
                    for idx, response in enumerate(item['responses'])
                }
                for future in as_completed(futures):
                    idx, extracted_answer, llm_answer_flag, llm_response_flag = future.result()
                    temp_answer[idx] = [extracted_answer, llm_answer_flag, llm_response_flag]
            for idx in range(len(temp_answer)):
                extracted_answer, llm_answer_flag, llm_response_flag = temp_answer[idx]
                item['extracted_answer'].append(extracted_answer)
                item['llm_answer_flag'].append(llm_answer_flag)
                item['llm_response_flag'].append(llm_response_flag)
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            f.flush()
    return data

def prm_evaluation_best_of_n(max_workers_num = 10, data_length = 5, critic_backbone = "tgi", critic_url = CRITIC_URL):
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    dataset_name = "gsm8k"

    input_file_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm2.jsonl'
    output_file_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3.jsonl'

    data = load_data(input_file_path, data_length)
    data = calculate_prm_values(data, critic_backbone, critic_url, max_workers_num, output_file_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")

    parser.add_argument('--max_workers_num', type=int, default=100, help='Maximum number of workers')
    parser.add_argument('--data_length', type=int, default=1319, help='Length of the data')
    parser.add_argument('--critic_backbone', type=str, default="tgi", help='Backbone for critic')
    parser.add_argument('--critic_url', type=str, default=TGI_URL, help='URL for critic backbone')
    
    args = parser.parse_args()

    prm_evaluation_best_of_n(
        max_workers_num=args.max_workers_num,
        data_length=args.data_length,
        critic_backbone=args.critic_backbone,
        critic_url=args.critic_url
    )

if __name__ == '__main__':
    main()