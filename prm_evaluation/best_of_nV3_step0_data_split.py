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


def load_and_split_data(input_file_path, data_length, num_splits):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp_data = json.loads(line)
            if temp_data.get('problem') is not None:
                temp_data['question'] = temp_data.get('problem', None)
            else:
                temp_data['question'] = temp_data.get('prompt', None)
            data.append(temp_data)
            if len(data) >= data_length:
                break

    split_data = []
    split_size = len(data) // num_splits
    for i in range(num_splits):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i < num_splits - 1 else len(data)
        split_data.append(data[start_index:end_index])

    return split_data

def prm_evaluation_best_of_n(id = 2, max_workers_num = 10, maxn = 5, data_length = 5, num_splits = 8, generate_backbone = "tgi", generate_url = TGI_URL, critic_backbone = "tgi", critic_url = CRITIC_URL):
    # project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    project_path = 'F://code//github//math-feedback//math-feedback//prm_evaluation//data//'
    input_file_path = project_path + 'test.jsonl'
    # output_file_dir = project_path + 'test/'
    output_file_dir = project_path + 'test//'


    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    output_file_path = project_path + 'test_' + str(id) + '.jsonl'
    
    split_data = load_and_split_data(input_file_path, data_length, num_splits)
    
    for idx, split in enumerate(split_data):
        output_file_path = project_path + f'test/test_{id}_part_{idx + 1}.jsonl'
        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            for item in split:
                out_file.write(json.dumps(item, ensure_ascii=False) + '\n')

    return

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")

    parser.add_argument('--id', type=str, default="500data", help='ID of the model')
    parser.add_argument('--max_workers_num', type=int, default=10, help='Maximum number of workers')
    parser.add_argument('--maxn', type=int, default=32, help='Maximum value of n')
    parser.add_argument('--data_length', type=int, default=500, help='Length of the data')
    parser.add_argument('--num_splits', type=int, default=8, help='Number of splits')
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