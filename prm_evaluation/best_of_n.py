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

from llm.llm_response import llm_response, TGI_URL, CRITIC_URL
from prm_inference.inference import get_scores, step_tag
from shepherd_prm.query_api import critic_math_problem, prepare_template

from prm_evaluation.openai.simple_evals.math_eval import QUERY_TEMPLATE, ANSWER_PATTERN, EQUALITY_TEMPLATE

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
    if response.count('\n') >= 2:
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

def get_best_of_n_reward_prod(question, response, backbone = "tgi", url = TGI_URL):
    '''
    get the reward product of the best response
    question: str, the question
    response: str, the response
    return: float, the reward product
    '''
    # insert the step tag
    output = response_insert_step_tag(response)
    # get the scores
    input_for_prm = f"{question} {output}"
    reward = get_scores(input_for_prm)
    # reward_mean = reward.mean().item() # mean
    reward_prod = reward.prod().item() # product
    return reward_prod

def best_of_n(n, question, responses = None, prm_value = None, backbone = "tgi", url = TGI_URL) -> str:
    '''
    get the best response from the n responses
    n: int, the number of responses
    question: str, the question
    responses: list, the responses
    prm_value: float, the prm value
    return: str, the best response
    '''
    best_reword_scores:float = float('-inf') # the best reword scores
    best_output = None # the best output

    if responses == None:
        responses = []
        for i in tqdm(range(n), desc="Generating I responses"):
            responses.append(llm_response(prompt=question, backbone=backbone, url=url))
    elif len(responses) > n:
        responses = responses[:n]

    for i in tqdm(range(n), desc="Calculating best response"):
        response = responses[i]
        if prm_value is not None and len(prm_value) >= i:
            reward_prod = prm_value[i]
        else:
            reward_prod = get_best_of_n_reward_prod(question, response, backbone, url)

        if reward_prod > best_reword_scores:
            best_reword_scores = reward_prod
            best_output = response
        
    return best_output

def problem_solved_by_best_of_n_with_referenceAnswerValue(n, data, backbone = "tgi", url = TGI_URL) -> float:
    '''
    solve the problem by the best of n, get accaracy
    n: int, the number of responses
    data: dict, the data, must contain the key 'question'
    return: float, the accaracy
    '''
    case_num = len(data)
    correct_num = 0
    for item in data:
        question = item['question']
        responses = item.get('responses', None)
        prm_value = item.get('prm_value', None)
        best_output = best_of_n(n, question, responses, prm_value, backbone, url)
        # get the answer
        match = re.search(ANSWER_PATTERN, best_output)
        extracted_answer = match.group(1) if match else None
        prompt = EQUALITY_TEMPLATE % {"expression1": extracted_answer, "expression2": item['answer']}
        for i in range(3):
            temp_response = llm_response(prompt, backbone, url)
            if temp_response.lower().strip() == "yes":
                correct_num += 1
                break

    return correct_num / case_num * 100

def problem_solved_by_best_of_n_with_referenceAnswer(n, data, backbone = "tgi", url = TGI_URL) -> float:
    '''
    solve the problem by the best of n, get accaracy
    n: int, the number of responses
    data: dict, the data, must contain the key 'answer' and 'question'
    return: float, the accaracy
    '''
    prompt_template_path = '/workspace/dujh22/math_feedback/shepherd_prm/templates/criticllm_math_template.txt'
    PROMPT_TEMPLATE = prepare_template(prompt_template_path)
    case_num = len(data)
    correct_num = 0
    for item in data:
        question = item['question']
        responses = item.get('responses', None)
        prm_value = item.get('prm_value', None)
        item['response'] = best_of_n(n, question, responses, prm_value, backbone, url)
        item = critic_math_problem(x=item, backbone=backbone, prompt_key="question", response_key="response", reference_key="answer", max_retry=3, PROMPT_TEMPLATE=PROMPT_TEMPLATE,  url=url) # get the critic result
        if item.get("critic_result") is not None:
            grade = float(item["critic_result"][0]["rating"])
            if grade >= 9:
                correct_num += 1

    return correct_num / case_num * 100

def paint_best_of_n_test():
    # Data: Assuming logarithmic x-axis and linear y-axis progression
    x = np.logspace(1, 3, 10)  # Generating points between 10^1 and 10^3
    y1 = np.linspace(70, 78, 10) + np.random.normal(0, 0.5, 10)  # Simulating 'Process-Supervised RM'
    y2 = np.linspace(72, 82, 10) + np.random.normal(0, 0.5, 10)  # Simulating 'Outcome-Supervised RM'
    y3 = np.linspace(66, 72, 10) + np.random.normal(0, 0.5, 10)  # Simulating 'Majority Voting'

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.semilogx(x, y1, marker='o', color='orange', label='Process-Supervised RM')
    ax.semilogx(x, y2, marker='o', color='blue', label='Outcome-Supervised RM')
    ax.semilogx(x, y3, marker='o', color='gray', label='Majority Voting')

    # Adding error bands as shaded regions
    ax.fill_between(x, y1-np.std(y1)*0.1, y1+np.std(y1)*0.1, color='orange', alpha=0.2)
    ax.fill_between(x, y2-np.std(y2)*0.1, y2+np.std(y2)*0.1, color='blue', alpha=0.2)
    ax.fill_between(x, y3-np.std(y3)*0.1, y3+np.std(y3)*0.1, color='gray', alpha=0.2)

    # Adding labels and title
    ax.set_xlabel('N = number of solutions per problem')
    ax.set_ylabel('% Problems Solved (Best-of-1860)')
    ax.set_title('Comparison of Resolution Methods')
    ax.legend()

    # Create a table below the axes
    raw_labels = ['% Solved (Best-of-1860)']
    cal_labels = ['ORM', 'PRM', 'Majority Voting']
    table_data = [['72.4', '78.2', '69.6']]
    table = plt.table(cellText=table_data, colLabels=cal_labels, 
                    rowLabels=raw_labels, cellLoc='center', loc='bottom', bbox=[0.25, -0.3, 0.5, 0.15])

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.4)

    # plt.show()
    plt.savefig('F://code//github//math-feedback//math-feedback//prm_evaluation//best_of_n.png', bbox_inches='tight')

def prm_evaluation_best_of_n(generate_backbone = "tgi", generate_url = TGI_URL, critic_backbone = "tgi", critic_url = CRITIC_URL):
    max_workers_num = 10
    n = [i for i in range(1, 64)] # n = 1, 2, 3, ..., 10

    # 0. Load data
    data_length = 100 # test
    data = []
    with open('/workspace/dujh22/math_feedback/prm_evaluation/data/test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            temp_data = json.loads(line.strip())
            temp_data['question'] = temp_data['problem'] # special solved because the data don't have the key 'question' but 'problem'
            data.append(temp_data)
            if len(data) >= data_length:
                break
    
    # 1. Generate n responses
    for item in tqdm(data, desc='Generating N Responses'):
        if item.get('responses') is None:
                item['responses'] = []
        maxn = n[-1] - len(item['responses'])
        if maxn > 0:
            with ThreadPoolExecutor(max_workers=max_workers_num) as executor:
                futures = {executor.submit(llm_response, QUERY_TEMPLATE.format(Question = item['question']), generate_backbone, generate_url): i for i in range(maxn)}             
                for future in as_completed(futures):
                    response = future.result()
                    item['responses'].append(response)
        elif maxn < 0:
            item['responses'] = item['responses'][:n[-1]]
        else:
            item['responses'] = item['responses']

    # 2. Calculate the PRM value
    for item in tqdm(data, desc='Calculating N PRMvalue'):
        if item.get('prm_value') is None:
                item['prm_value'] = []
        maxn = n[-1] - len(item['prm_value'])
        if maxn > 0:
            with ThreadPoolExecutor(max_workers=max_workers_num) as executor:
                futures = {executor.submit(get_best_of_n_reward_prod, item['question'], response, generate_backbone, generate_url): response for response in item['responses'][len(item['prm_value']):n[-1]]}
                for future in as_completed(futures):
                    response = future.result()
                    item['prm_value'].append(response)
        elif maxn < 0:
            item['prm_value'] = item['prm_value'][:n[-1]]
        else:
            item['prm_value'] = item['prm_value']

    accuracy_with_referenceAnswer = []
    accuracy_with_referenceAnswerValue = []

    # 3. Calculate the accuracy with referenceAnswer
    with ThreadPoolExecutor(max_workers=max_workers_num) as executor:
        futures = {executor.submit(problem_solved_by_best_of_n_with_referenceAnswer, ni, data, critic_backbone, critic_url): ni for ni in n}
    
        with tqdm(total=len(futures), desc='problem_solved_by_best_of_n_with_referenceAnswer') as progress:
            for future in as_completed(futures):
                result = future.result()
                accuracy_with_referenceAnswer.append(result)
                progress.update(1)  # 更新进度条

    # 4. Calculate the accuracy with referenceAnswerValue
    with ThreadPoolExecutor(max_workers=max_workers_num) as executor:
        futures = {executor.submit(problem_solved_by_best_of_n_with_referenceAnswerValue, ni, data, generate_backbone, generate_url): ni for ni in n}
    
        with tqdm(total=len(futures), desc='problem_solved_by_best_of_n_with_referenceAnswerValue') as progress:
            for future in as_completed(futures):
                result = future.result()
                accuracy_with_referenceAnswerValue.append(result)
                progress.update(1)  # 
                
    # output data to csv
    with open('/workspace/dujh22/math_feedback/prm_evaluation/data/test2.csv', 'w', encoding='utf-8') as f:
        f.write('n,accuracy_with_referenceAnswer,accuracy_with_referenceAnswerValue\n')
        for i in range(len(n)):
            f.write(f"{n[i]},{accuracy_with_referenceAnswer[i]},{accuracy_with_referenceAnswerValue[i]}\n")

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.semilogx(n, accuracy_with_referenceAnswer, marker='o', color='orange', label='accuracy_with_referenceAnswer')
    ax.semilogx(n, accuracy_with_referenceAnswerValue, marker='o', color='blue', label='accuracy_with_referenceAnswerValue')

    # Adding error bands as shaded regions
    ax.fill_between(n, accuracy_with_referenceAnswer-np.std(accuracy_with_referenceAnswer)*0.1, accuracy_with_referenceAnswer+np.std(accuracy_with_referenceAnswer)*0.1, color='orange', alpha=0.2)
    ax.fill_between(n, accuracy_with_referenceAnswerValue-np.std(accuracy_with_referenceAnswerValue)*0.1, accuracy_with_referenceAnswerValue+np.std(accuracy_with_referenceAnswerValue)*0.1, color='blue', alpha=0.2)
    
    # Adding labels and title
    ax.set_xlabel('N = number of solutions per problem')
    ax.set_ylabel('% Problems Solved (Best-of-1860)')
    ax.set_title('Comparison of Resolution Methods')
    ax.legend()

    # plt.show()
    plt.savefig('/workspace/dujh22/math_feedback/prm_evaluation/best_of_n2.png', bbox_inches='tight') 

    # 保存结果
    with open('/workspace/dujh22/math_feedback/prm_evaluation/data/test2.jsonl', 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    prm_evaluation_best_of_n()

if __name__ == '__main__':
    main()