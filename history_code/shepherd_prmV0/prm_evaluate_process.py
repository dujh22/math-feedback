import json
import argparse
from functools import partial
import re
import requests
import random

from query_api import build_training_file, critic_math_problem, prepare_template


def query_tgi_completion(prompt):
    url = "http://xxx:8080/generate"
    configs = [
        {"temperature": 0.1, "top_p": 0.7},
        {"temperature": 0.9, "top_p": 0.9},
    ]
    if random.randint(0, 5) == 0:
        config = configs[0]
    else:
        config = configs[1]
        
    payload = {
        "best_of": 1,
        "decoder_input_details": False,
        "details": False,
        "do_sample": True,
        "max_new_tokens": 2048,
        "seed": random.randint(0, 100000),
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"]
  }
    requests.post(url, json=payload, verify=False)


def split_response(response):
    steps = re.split(r"\n", response)
    return steps
    

def generate_process(x, prompt_key, response_key, num_path=3, backbone="glm-code-v3"):
    prompt = x[prompt_key]
    response = x[response_key]
    output = []
    steps = split_response(response)
    for idx in range(len(steps)):
        extension_path = []        
        for _p in range(num_path):
            _step = "\n".join(steps[:idx+1])
            query_prompt = f"<|user|>\n{prompt}<|assistant|>\n{_step}"
        
            result = None
            for _ in range(3):
                try:
                    result = query_tgi_completion(query_prompt)
                    if result is not None:
                        break
                except Exception as e:
                    continue
            if result is None:
                continue
            
            extension_path.append(result)

        output.append({
            "step": _step,
            "extension": extension_path
        })

    x["generated_paths"] = output
    return x


def evaluate_process(x, prompt_key="prompt", process_response_key="generated_paths", reference_answewr_key="reference", max_retry=3):
    generated_paths = x[process_response_key]

    for path in generated_paths:
        step_paths = path["extension"]
        ratings = []

        for step_path in step_paths:
            temp_item = {
                prompt_key: x[prompt_key],
                "response": step_path,
                reference_answewr_key: x[reference_answewr_key]
            }
            result = critic_math_problem(
                temp_item,
                backbone="chatglm_platform", 
                prompt_key=prompt_key,
                response_key="response",
                reference_key=reference_answewr_key
            )
            rating = result["critic_result"][0]["rating"]
            ratings.append(rating)

        path["ratings"] = ratings
        ratings_binary = [1 if x >= 8 else 0 for x in ratings]
        path["soft_label"] = sum(ratings_binary) / len(ratings_binary)
        path["hard_label"] = 1 if path["soft_label"] >= 0.5 else 0

    x[process_response_key] = generated_paths
    return x


def select_math_data_by_rating(input_file):
    if isinstance(input_file, str):
        data = [json.loads(x) for x in open(input_file)]
    else:
        data = input_file

    def judge_scores(scores, lower_bound=7):
        avg_score = sum(scores) / len(scores)
        above_bound = [1 if x >= lower_bound else 0 for x in scores]
        return avg_score, sum(above_bound) / len(above_bound)
    
    def func(x, lower_bound=8):
        results = x["critic_result"]
        if len(results) == 0:
            return None
        ratings = [item["rating"] for item in results if isinstance(item["rating"], str)]
        # print(ratings)
        ratings = [float(x) for x in ratings]
        avg_score, pass_rate = judge_scores(ratings, lower_bound=lower_bound)
        x["critic_scores"] = {
            "ratings": ratings,
            "avg_score": avg_score,
            "pass_rate": pass_rate
        }
        return x
    
    processed = [func(x) for x in data]
    return processed
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="response")
    parser.add_argument("--backbone", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--prompt_key", type=str, default=None)
    # "gpt-4-1106-preview"
    parser.add_argument("--skip_response", action="store_true", default=False)
    parser.add_argument("--skip_generated", action="store_true", default=False)
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--reference_key", type=str, default="answer")
    parser.add_argument("--response_key", type=str, default="response")
    args = parser.parse_args()
    
    if args.mode == "generation":
        build_training_file(
            input_file=args.input_file,
            output_file=args.input_file.replace(".jsonl", "_path.jsonl"),
            worker_func=partial(
                generate_process, 
                prompt_key=args.prompt_key, 
                response_key=args.response_key
            ),
            is_glm=False
        )
    elif args.mode == "critic":
        build_training_file(
            input_file=args.input_file,
            output_file=args.input_file.replace(".jsonl", "_math_critic.jsonl"),
            worker_func=partial(critic_math_problem, backbone=args.backbone, prompt_key=args.prompt_key, response_key=args.response_key, reference_key=args.reference_key),
            is_glm=False
        )
