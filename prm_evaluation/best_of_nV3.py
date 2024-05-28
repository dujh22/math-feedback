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

def response_insert_step_tag(response:str) -> str:
    '''
    insert the step tag into the response
    response: str, the response
    return: str, the response with step tag
    '''
    # first split the response by \n\n
    if response.count('\n\n') >= 2:
        steps = re.split(r"\n\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # delete space
        separator = f" {step_tag}\n\n"
        new_response = separator.join(steps)
        new_response = new_response + ' ' + step_tag
        return new_response
    # then split the response by \n
    elif response.count('\n') >= 2:
        steps = re.split(r"\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0]
        separator = f" {step_tag}\n"
        new_response = separator.join(steps)
        new_response = new_response + ' ' + step_tag
        return new_response
    # otherwise split the response by .
    else:
        # notice that the response may contain '。' instead of '.'
        notice_character = '.'
        if '。' in response:
            notice_character = '。'
            steps = re.split(r'。', response)
        else:
            steps = re.split(r'(?<=[^.0-9])\.(?=[^0-9])', response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        separator = f" {step_tag}{notice_character}"
        new_response = separator.join(steps)
        new_response = new_response + ' ' + step_tag
        return new_response

def get_best_of_n_reward_prod(question, response, answer, backbone, url, id):
    # 1.prm
    # insert the step tag
    output = response_insert_step_tag(response)
    # get the scores
    input_for_prm = f"{question} {output}"
    print(f"1.get {id} reward--------------------------")
    reward = get_scores(input_for_prm)
    # reward_mean = reward.mean().item() # mean
    reward_prod = reward.prod().item() # product

    # 2.llm_answer
    # get the answer
    match = re.search(ANSWER_PATTERN, response)
    extracted_answer = match.group(1) if match else None
    prompt = EQUALITY_TEMPLATE % {"expression1": extracted_answer, "expression2": answer}
    llm_answer_flag = 0
    print(f"2.get {id} response answer judgements-----------")
    for i in range(3):
        temp_response = llm_response(prompt, backbone, url)
        if temp_response.lower().strip() == "yes":
            llm_answer_flag = 1
            break

    # 3.llm_response
    print(f"3.get {id} response judgements-----------")
    item = {"question": question, "response": response, "answer": answer}
    item = critic_math_problem(x=item, backbone=backbone, prompt_key="question", response_key="response", reference_key="answer", max_retry=3, PROMPT_TEMPLATE=PROMPT_TEMPLATE,  url=url) # get the critic result
    llm_response_flag = 0
    if item.get("critic_result") is not None:
        grade = float(item["critic_result"][0]["rating"])
        if grade >= 9:
           llm_response_flag = 1

    return reward_prod, extracted_answer, llm_answer_flag, llm_response_flag


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

def generate_responses(data, maxn, generate_backbone, generate_url, max_workers_num, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc='Generating N Responses'):
            item['responses'] = []
            if generate_backbone == 'mistral7b':
                for _ in tqdm(range(maxn), desc="Generating responses with mistral7b"):
                    item['responses'].append(get_mistral_response(item['question']))
            else:
                with ThreadPoolExecutor(max_workers=max_workers_num) as executor:
                    futures = {executor.submit(llm_response, QUERY_TEMPLATE.format(Question=item['question']), generate_backbone, generate_url): i for i in range(1, maxn + 1)}
                    for future in as_completed(futures):
                        response = future.result()
                        item['responses'].append(response)
            # 写出每个item到文件
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return data

def calculate_prm_values(data, critic_backbone, critic_url, max_workers_num, output_file_path):
    id = 0
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc='Calculating N PRMvalue'):
            id += 1
            item['prm_value'] = []
            item['extracted_answer'] = []
            item['llm_answer_flag'] = []
            item['llm_response_flag'] = []
            with ThreadPoolExecutor(max_workers=max_workers_num) as executor:
                futures = {executor.submit(get_best_of_n_reward_prod, item['question'], response, item['answer'], critic_backbone, critic_url, id): response for response in item['responses']}
                for future in as_completed(futures):
                    reward_prod, extracted_answer, llm_answer_flag, llm_response_flag = future.result()
                    item['prm_value'].append(reward_prod)
                    item['extracted_answer'].append(extracted_answer)
                    item['llm_answer_flag'].append(llm_answer_flag)
                    item['llm_response_flag'].append(llm_response_flag)
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
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

def prm_evaluation_best_of_n(id = 2, max_workers_num = 10, maxn = 5, data_length = 5,generate_backbone = "tgi", generate_url = TGI_URL, critic_backbone = "tgi", critic_url = CRITIC_URL):
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    input_file_path = project_path + 'test.jsonl'
    output_file_path = project_path + 'test_' + str(id) + '.jsonl'
    output_csv_path = project_path + 'test_' + str(id) + '.csv'
    output_pic_path = project_path + 'test_' + str(id) + '.png'
    # max_workers_num = 10
    # maxn = 5
    # data_length = 5 # test
    # generate_backbone = "mistral7b"
    # generate_url = TGI_URL

    N = [i for i in range(1, maxn + 1)]
    
    data = load_data(input_file_path, data_length)

    if data[0].get('responses') is None:
        data = generate_responses(data, maxn, generate_backbone, generate_url, max_workers_num, output_file_path)

    if data[0].get('prm_value') is None:
        data = calculate_prm_values(data, critic_backbone, critic_url, max_workers_num, output_file_path)

    calculate_accuracy_and_plot(data, N, output_csv_path, output_pic_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")

    parser.add_argument('--id', type=str, default="100data", help='ID of the model')
    parser.add_argument('--max_workers_num', type=int, default=10, help='Maximum number of workers')
    parser.add_argument('--maxn', type=int, default=32, help='Maximum value of n')
    parser.add_argument('--data_length', type=int, default=100, help='Length of the data')
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