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
import wandb
from datetime import datetime

from llm.llm_response import llm_response, TGI_URL, CRITIC_URL
from prm_inference.inference import get_scores, step_tag
from shepherd_prm.query_api import critic_math_problem, prepare_template

from prm_evaluation.openai.simple_evals.math_eval import QUERY_TEMPLATE, ANSWER_PATTERN, EQUALITY_TEMPLATE
from prm_inference.inference_mistral7b import get_mistral_response

PROMPT_TEMPLATE = prepare_template('/workspace/dujh22/math_feedback/shepherd_prm/templates/criticllm_math_template.txt')

# 初始化 wandb
wandb.login(key="76ea5b2b06f6f9a718116bb3ec0bd54936f2fded")
wandb.init(
    project="prm_evaluation",
    name="rm_%s" % datetime.now().strftime("%m%dT%H:%M")
)

def load_data(input_file_path, data_length):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp_data = json.loads(line.strip())
            data.append(temp_data)
            if len(data) >= data_length:
                break
    return data

def calculate_accuracy_and_plot(data, N, output_csv_path, output_pic_path):
    accuracy_with_referenceAnswer = []
    accuracy_with_referenceAnswerValue = []

    with open(output_csv_path, 'w', encoding='utf-8') as f:
        f.write('n,accuracy_with_referenceAnswer,accuracy_with_referenceAnswerValue\n')
        for ni in tqdm(N, desc='Calculating Accuracy with referenceAnswer & referenceAnswerValue'):
            llm_answer_flag_correct_num = 0
            llm_response_flag_correct_num = 0
            for item in data:
                best_prm_value_id = item['prm_value'][:ni].index(max(item['prm_value'][:ni]))
                if item['llm_answer_flag'][best_prm_value_id] == 1:
                    llm_answer_flag_correct_num += 1
                if item['llm_response_flag'][best_prm_value_id] == 1:
                    llm_response_flag_correct_num += 1
            accuracy_with_referenceAnswerValue.append(llm_answer_flag_correct_num / len(data) * 100)
            accuracy_with_referenceAnswer.append(llm_response_flag_correct_num / len(data) * 100)
            f.write(f"{ni},{accuracy_with_referenceAnswer[-1]},{accuracy_with_referenceAnswerValue[-1]}\n")
            wandb.log({"n": ni, "accuracy_with_referenceAnswer": accuracy_with_referenceAnswer[-1], "accuracy_with_referenceAnswerValue": accuracy_with_referenceAnswerValue[-1]})
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.semilogx(N, accuracy_with_referenceAnswer, marker='o', color='orange', label='accuracy_with_referenceAnswer')
    ax.semilogx(N, accuracy_with_referenceAnswerValue, marker='o', color='blue', label='accuracy_with_referenceAnswerValue')

    ax.fill_between(N, accuracy_with_referenceAnswer-np.std(accuracy_with_referenceAnswer)*0.1, accuracy_with_referenceAnswer+np.std(accuracy_with_referenceAnswer)*0.1, color='orange', alpha=0.2)
    ax.fill_between(N, accuracy_with_referenceAnswerValue-np.std(accuracy_with_referenceAnswerValue)*0.1, accuracy_with_referenceAnswerValue+np.std(accuracy_with_referenceAnswerValue)*0.1, color='blue', alpha=0.2)

    degree = 3
    z1 = np.polyfit(N, accuracy_with_referenceAnswer, degree)
    p1 = np.poly1d(z1)
    ax.plot(N, p1(N), linestyle='--', color='red', label='Trendline accuracy_with_referenceAnswer')

    z2 = np.polyfit(N, accuracy_with_referenceAnswerValue, degree)
    p2 = np.poly1d(z2)
    ax.plot(N, p2(N), linestyle='--', color='black', label='Trendline accuracy_with_referenceAnswerValue')

    ax.set_xlabel('N = number of solutions per problem')
    ax.set_ylabel('% Problems Solved (Best-of-N)')
    ax.set_title('Comparison of Resolution Methods')
    ax.legend()

    plt.savefig(output_pic_path, bbox_inches='tight')
    wandb.log({"plot": wandb.Image(output_pic_path)})

def prm_evaluation_best_of_n(maxn = 5, data_length = 5):
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/'
    input_file_path = project_path + 'test_rm3_mathshepherd_prm.jsonl'
    output_csv_path = project_path + 'test_rm3_mathshepherd_prm.csv'
    output_pic_path = project_path + 'test_rm3_mathshepherd_prm.png'

    N = [i for i in range(1, maxn + 1)]
    
    data = load_data(input_file_path, data_length)
    calculate_accuracy_and_plot(data, N, output_csv_path, output_pic_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")

    parser.add_argument('--maxn', type=int, default=32, help='Maximum value of n')
    parser.add_argument('--data_length', type=int, default=100, help='Length of the data')
    
    args = parser.parse_args()

    prm_evaluation_best_of_n(
        maxn=args.maxn,
        data_length=args.data_length,
    )

if __name__ == '__main__':
    main()