import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer 
from transformers import AutoModelForCausalLM  
import torch  

good_token = '+'  # define positive token
bad_token = '-'  # define negative token
step_tag = 'ки'  # define step tag
model_path = '/workspace/dujh22/models/math-shepherd-mistral-7b-prm'  

tokenizer = AutoTokenizer.from_pretrained(model_path)
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] 
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  

devices = [torch.device(f"cuda:{i}") for i in range(8)]
models = [AutoModelForCausalLM.from_pretrained(model_path).to(devices[i]).eval() for i in range(8)]

def get_scores(input_for_prm: str, model_index: int) -> torch.Tensor:
    device = devices[model_index]
    model = models[model_index]
    input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)  
    with torch.no_grad():  
        logits = model(input_id).logits[:,:,candidate_tokens]  
        scores = logits.softmax(dim=-1)[:,:,0]  
        step_scores = scores[input_id == step_tag_id]      
        # score = step_scores.prod().item()
        score = step_scores.mean().item()
    return score

def load_data(input_file_path, data_length):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp_data = json.loads(line.strip())
            data.append(temp_data)
            if len(data) >= data_length:
                break
    return data

def calculate_prm_values(data, output_file_path):
    def process_response(item, i, j):
        input_for_prm = f"{item['question']} {item['responses'][i * 8 + j]}"
        return i * 8 + j, get_scores(input_for_prm, j)

    id = 0
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc='Calculating N PRMvalue'):
            prm_dict = {}
            futures = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                for i in range(len(item['responses']) // 8):
                    for j in range(8):
                        futures.append(executor.submit(process_response, item, i, j))
                for future in as_completed(futures):
                    idx, score = future.result()
                    prm_dict[idx] = score
            prm = [prm_dict[idx] for idx in sorted(prm_dict)]
            item['prm_value'] = prm
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            f.flush()
    return data

def prm_evaluation_best_of_n(data_length = 5):
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/'
    input_file_path = project_path + 'test_rm3.jsonl'
    output_file_path = project_path + 'test_rm3_mathshepherd_prm.jsonl'

    data = load_data(input_file_path, data_length)
    data = calculate_prm_values(data, output_file_path)

def main():
    prm_evaluation_best_of_n(data_length=500)

if __name__ == '__main__':
    main()