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


def data_preprocess(data, dataset_name):
    if dataset_name == "math500":
        if data.get('problem') is not None:
            data['question'] = data.get('problem', None)
        else:
            data['question'] = data.get('prompt', None)
        data['dataset'] = dataset_name
        return data
    elif dataset_name == "gsm8k":
        data['solution'] = data.get('answer', None)
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        INVALID_ANS = "[invalid]"
        match = ANS_RE.search(data['solution'])
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            data['answer'] = match_str
        else:
            data['answer'] = INVALID_ANS
        return data
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_and_split_data(input_file_path, data_length, num_splits, dataset_name):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for id, line in enumerate(f):
            temp_data = json.loads(line)
            temp_data2 = data_preprocess(temp_data, dataset_name)
            temp_data2['id'] = id
            data.append(temp_data2)
            if len(data) >= data_length:
                break

    split_data = []
    split_size = len(data) // num_splits
    for i in range(num_splits):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i < num_splits - 1 else len(data)
        split_data.append(data[start_index:end_index])

    return split_data

def prm_evaluation_best_of_n(data_length = 5, num_splits = 8, dataset_name = "math500"):
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    input_file_path = project_path + dataset_name + '.jsonl'
    output_file_dir = project_path + dataset_name + '/'

    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    
    split_data = load_and_split_data(input_file_path, data_length, num_splits, dataset_name)
    
    for idx, split in enumerate(split_data):
        output_file_path = output_file_dir + f'{dataset_name}_part_{idx + 1}.jsonl'
        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            for item in split:
                out_file.write(json.dumps(item, ensure_ascii=False) + '\n')

    return

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")

    parser.add_argument('--data_length', type=int, default=1319, help='Length of the data')
    parser.add_argument('--num_splits', type=int, default=8, help='Number of splits')
    parser.add_argument('--dataset_name', type=str, default='gsm8k', help='Name of the dataset')
    
    args = parser.parse_args()

    prm_evaluation_best_of_n(
        data_length=args.data_length,
        num_splits=args.num_splits,
        dataset_name=args.dataset_name
    )

if __name__ == '__main__':
    main()