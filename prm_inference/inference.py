from transformers import AutoTokenizer 
from transformers import AutoModelForCausalLM  
import torch  

good_token = '+'  # define positive token
bad_token = '-'  # define negative token
step_tag = 'ки'  # define step tag

# model_path = 'F:/code/github/math-feedback/math-feedback/prm_inference/models/peiyi9979_math_shepherd_mistral_7b_prm'
model_path = '/workspace/dujh22/models/math-shepherd-mistral-7b-prm'  # define model path

tokenizer = AutoTokenizer.from_pretrained(model_path)
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] 
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  
model = AutoModelForCausalLM.from_pretrained(model_path).eval()

def get_scores(input_for_prm: str) -> torch.Tensor:
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])  
    with torch.no_grad():  
        logits = model(input_id).logits[:,:,candidate_tokens]  
        scores = logits.softmax(dim=-1)[:,:,0]  
        step_scores = scores[input_id == step_tag_id]        
    return step_scores


def main():
    question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки""" # 18 is right
    output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки""" # 17 is wrong

    for output in [output1, output2]:
        input_for_prm = f"{question} {output}"  # combine question and output
        scores = get_scores(input_for_prm)
        print(scores)

    # 输出：      
    # tensor([0.9955, 0.9958, 0.9983, 0.9957])
    # tensor([0.9955, 0.9958, 0.9983, 0.0240]) 

if __name__ == '__main__':
    main()
