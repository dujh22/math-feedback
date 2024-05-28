import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import numpy as np
import argparse

from datetime import datetime

from llm.llm_response import llm_response, TGI_URL, CRITIC_URL
from prm_evaluation.openai.simple_evals.math_eval import QUERY_TEMPLATE

gpu_id = 0

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device = torch.device("cuda:" + str(gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
model_path = "/workspace/dujh22/models/mistral-7B-sft"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

def get_mistral_response(question):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_data(input_file_path, data_length):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp_data = json.loads(line.strip())
            if temp_data.get('question') is None:
                if temp_data.get('problem') is not None:
                    temp_data['question'] = temp_data.get('problem', None)
                else:
                    temp_data['question'] = temp_data.get('prompt', None)
            data.append(temp_data)
            if len(data) >= data_length:
                break
    return data

def get_existing_ids(output_file_path):
    existing_ids = set()
    try:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                existing_ids.add(item['unique_id'])
    except FileNotFoundError:
        pass
    return existing_ids

def generate_responses(data, maxn, generate_backbone, generate_url, max_workers_num, output_file_path):
    existing_ids = get_existing_ids(output_file_path)

    with open(output_file_path, 'a', encoding='utf-8') as f:
        for item in tqdm(data, desc='Generating N Responses'):
            if item['unique_id'] in existing_ids:
                continue

            item['responses'] = []
            if generate_backbone == 'mistral7b':
                for _ in tqdm(range(maxn), desc="Generating responses with mistral7b"):
                    item['responses'].append(get_mistral_response(QUERY_TEMPLATE.format(Question=item['question'])))
            else:
                with ThreadPoolExecutor(max_workers=max_workers_num) as executor:
                    futures = {executor.submit(llm_response, QUERY_TEMPLATE.format(Question=item['question']), generate_backbone, generate_url): i for i in range(1, maxn + 1)}
                    for future in as_completed(futures):
                        response = future.result()
                        item['responses'].append(response)
            # 写出每个item到文件
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            f.flush()  # 确保数据被及时写入文件
    return data

def prm_evaluation_best_of_n(id = 2, max_workers_num = 10, maxn = 5, data_length = 5,generate_backbone = "tgi", generate_url = TGI_URL, critic_backbone = "tgi", critic_url = CRITIC_URL):
    idx = gpu_id
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    input_file_path = project_path + f'test/test_{id}_part_{idx + 1}.jsonl'
    
    output_file_dir = project_path + 'test1/'
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    output_file_path = project_path + f'test1/test_{id}_part_{idx + 1}.jsonl'
    
    data = load_data(input_file_path, data_length)
    data = generate_responses(data, maxn, generate_backbone, generate_url, max_workers_num, output_file_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")

    parser.add_argument('--id', type=str, default="500data", help='ID of the model')
    parser.add_argument('--max_workers_num', type=int, default=10, help='Maximum number of workers')
    parser.add_argument('--maxn', type=int, default=32, help='Maximum value of n')
    parser.add_argument('--data_length', type=int, default=500, help='Length of the data')
    parser.add_argument('--generate_backbone', type=str, default="mistral7b", help='Backbone for generation')
    parser.add_argument('--generate_url', type=str, default=TGI_URL, help='URL for generation backbone')
    parser.add_argument('--critic_backbone', type=str, default="tgi", help='Backbone for critic')
    parser.add_argument('--critic_url', type=str, default=TGI_URL, help='URL for critic backbone')
    
    args = parser.parse_args()

    prm_evaluation_best_of_n(
        id=args.id,
        max_workers_num=args.max_workers_num,
        maxn=args.maxn,
        data_length=args.data_length,
        generate_backbone=args.generate_backbone,
        generate_url=args.generate_url,
        critic_backbone=args.critic_backbone,
        critic_url=args.critic_url
    )

if __name__ == '__main__':
    main()