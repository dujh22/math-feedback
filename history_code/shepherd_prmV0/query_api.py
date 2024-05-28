import argparse
import time
import json
import random
import requests
from tqdm import tqdm
from functools import partial
from multiprocess import Pool, Queue, Process
import warnings
import openai
import re

random.seed(666)
warnings.filterwarnings("ignore")

NUM_PROCESS=10

QUEUE_SIZE=1000000


TEMPERATURE = 0.9
TOPP = 0.2
PROMPT_TEMPLATE = None


def query_chatglm_platform(prompt, history=[], do_sample=True, max_tokens=2048):
    url = "http://xxx:9090/v1/chat/completions"

    messages = []
    for turn in history:
        messages.append({
            "role": "user",
            "content": turn["prompt"],
        })
        messages.append({
            "role": "assistant",
            "content": turn["response"],
        })
    messages.append({
        "role": "user",
        "content": prompt,
    })

    payload = {
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOPP,
        # "model": self.model_version,
        "max_tokens": max_tokens,
        "do_sample": do_sample,
        "stream": False,
        "seed": random.randint(1, 10000000),
    }

    # response = requests.post(self.url, data=payload, headers=self.headers, verify=False)
    response = requests.post(url, json=payload, verify=False)
    
    if response.status_code == 200:
        answer = json.loads(response.text)
        # print(answer)
        # if answer["choices"][0]["finish_reason"] != "eos_token":
            # answer = None
        # else:
        answer = answer["choices"][0]["message"]["content"]
    else:
        print(response.text)
        answer = None

    return answer


def query_chatglm_tgi(prompt, history=[], do_sample=True, max_tokens=2048, max_retry=3):
    url = "http://xxx:8080/generate"
    messages = ""
    for turn in history:
        ques, ans = turn["prompt"], turn["response"]
        messages += f"<|user|>\n{ques}<|assistant|>\n{ans}"

    messages += f"<|user|>\n{prompt}<|assistant|>\n"
    inputs = {
        "inputs": messages,
        "stream": False,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": True,
            "details": False,
            "do_sample": do_sample,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "seed": None,
            "temperature": 1,
            "top_p": 0.9,
            "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"]
        }
    }
   
    for _ in range(max_retry):
        output = requests.post(url, json=inputs)
        if output.status_code == 200:
            output = json.loads(output.text)
            # results.append(output[0]["generated_text"])
            result = output["generated_text"]
            break
        else:
            print(output.text)   
    else:
        result = None

    return result


def query_gpt4(prompt, history=[], backbone="gpt-3.5-turbo"):
    messages = []

    for turn in history:
        messages.append({"role": "user", "content": turn["prompt"]})
        messages.append({"role": "assistant", "content": turn["response"]})
    messages.append({"role": "user", "content": prompt})
    # create a chat completion
    chat_completion = openai.ChatCompletion.create(
                            model=backbone,
                            messages=messages,
                            temperature=TEMPERATURE,
                            top_p=TOPP,
                            # max_tokens=2048,
                        )
    return chat_completion.choices[0].message.content


def query_gpt4_with_standard_format(inputs):
    messages = inputs["messages"]

    chat_completion = openai.ChatCompletion.create(
                            model="gpt-4-1106-preview",
                            messages=messages,
                            temperature=TEMPERATURE,
                            top_p=TOPP,
                            # max_tokens=2048,
                        )
    return chat_completion.choices[0].message.content


def worker_build_training_pair(task_queue, done_queue, worker_func, max_retry=3, is_glm=False):
    for line in iter(task_queue.get, "STOP"):
        item = json.loads(line)
        response = None
        for _ in range(max_retry):
            try:
                response = worker_func(item)
            except Exception as e:
                print("error:", e)
                # exit()
                continue

            if response is not None:
                break
            
            time.sleep(3)
        else:
            continue

        done_queue.put(item)

    done_queue.put("COMPLETE")


def build_training_file(input_file, output_file, worker_func, is_glm=False, num_process=None):
    if num_process is None:
        num_processes = NUM_PROCESS
    else:
        num_processes = num_process
        
    task_queue, done_queue = Queue(maxsize=QUEUE_SIZE), Queue(maxsize=QUEUE_SIZE)

    def read_data_into_queue():
        cnt = 0
        
        with open(input_file, "r") as r:
            print("read files")
            for line in r:
                task_queue.put(line)
                cnt += 1
                # cnt -= 1
                # if cnt <= 0:
                    # break
            print("read files done: ", cnt)


        for _ in range(num_processes):
            task_queue.put('STOP')

    processes = []
    for _ in range(num_processes):
        process = Process(target=partial(worker_build_training_pair, is_glm=is_glm),
                    args=(task_queue, done_queue, worker_func))
        process.start()
        processes.append(process)

    process = Process(target=read_data_into_queue)
    process.start()

    progress_bar = tqdm()
    print("----- GOGOGOGOGOGOGO !!!!!")
    with open(output_file, 'w') as w:
        num_finished = 0
        num_save = 0
        while num_finished < num_processes:
            item = done_queue.get()
            if item == 'COMPLETE':
                num_finished += 1
            else:
                w.write(json.dumps(item, ensure_ascii=False) + '\n')
                w.flush()
                num_save += 1
                # print(f'save {num_save} examples to {output_file}', end='\r')
                progress_bar.update(1)


def standard_prompt_response(
    x, 
    response_key="response", 
    skip_response=False, 
    skip_generated=False, 
    backbone="gpt-3.5-turbo", 
    prompt_key="prompt", 
    num_generation=1
):

    if skip_response and response_key in x:
        print("skip")
        return x[response_key]
    
    # if skip_generated and response_key in x:
        # return x["gpt4_turbo_response"]
        
    # if "messages" in x:    
    #     raise NotImplementedError 
    #     result = query_gpt4_with_standard_format(x)
    #     x["messages"].append(
    #         {"role": "assistant", "content": result}
    #     )
    #     if "gpt4_response" in x:
    #         x.pop("gpt4_response")
    #     x["gpt4_turbo_response"] = result
    #     # question = x['messages'][-2]["content"]
    #     x["sythetic_prompt"] = extract(result)
    # else:
    if "history" in x:
        history = x["history"]
    else:
        history = []
        
    prompt = x[prompt_key]

    responses = []
    for i in range(num_generation):
        max_try = 3
        for _ in range(max_try):
            if backbone == "chatglm_platform":
                result = query_chatglm_platform(prompt, history)
            elif backbone == "tgi":
                result = query_chatglm_tgi(prompt, history)
            elif backbone == "chatglm_ipo":
                # result = query_chatglm(prompt, history)
                raise NotImplementedError
            elif "gpt" in backbone:
                result = query_gpt4(prompt, history, backbone=backbone)
            else:
                raise NotImplementedError

            if result is None:
                continue
        if result is not None:
            responses.append((f"reply_{i}", result))

    import random

    if len(result) > 0:        
        rnm = random.randint(0, 20)
        if rnm == 0:
            print(f"#### Question: {prompt} ------ \n Response: ", result[0])        # print("#### Original response: ", result)
            print()

    if num_generation == 1:
        result = result[0][1]

    x[response_key] = result
    return result


def critic_math_problem(x, backbone="chatglm_platform", prompt_key="prompt", response_key="response", reference_key="answer", max_retry=3):
    # prompt = 
    response = x[response_key]
    if isinstance(response, str):
        response = [response]
    prompt = x[prompt_key]
    
    outputs = []
    for resp_item in response:
        rating = None
        if isinstance(resp_item, str):
            resp = resp_item
        elif isinstance(resp_item, list) or isinstance(resp_item, tuple):
            resp = resp_item[-1]
        else:
            raise NotImplementedError

        for _ in range(max_retry):
            input_data = PROMPT_TEMPLATE.format(
                problem=prompt,
                reference_answer=x[reference_key],
                assistant_ansiwer=resp
            )    

            if backbone == "chatglm_platform":
                result = query_chatglm_platform(input_data)
            elif backbone == "tgi":
                result = query_chatglm_tgi(input_data)
            elif backbone == "chatglm_ipo":
                # result = query_chatglm(input_data)
                raise NotImplementedError
            else:
                raise NotImplementedError

            rating = re.findall(r"\[\[(\d+)\]\]", result)

            if len(rating) == 0:
                continue
            else:
                rating = rating[0]
                break
            
        if rating is not None:
            outputs.append({
                "response": resp,
                "rating": rating,
                "judge_result": result
            })
    x["critic_result"] = outputs
    return x
    

def prepare_template(prompt_filepath):
    print(f"Load prompt template from {prompt_filepath}...")
    global PROMPT_TEMPLATE
    PROMPT_TEMPLATE = open(prompt_filepath).read().strip()


if __name__ == '__main__':
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
    parser.add_argument("--num_generation", type=str, default=1)
    parser.add_argument("--num_process", type=int, default=10)
    args = parser.parse_args()
    
    if args.mode == "critic":
        prepare_template(args.prompt_template)
        build_training_file(
            input_file=args.input_file,
            output_file=args.input_file.replace(".jsonl", "_math_critic.jsonl"),
            worker_func=partial(
                critic_math_problem, 
                backbone=args.backbone, 
                prompt_key=args.prompt_key, 
                reference_key=args.reference_key,
                response_key=args.response_key
            ),
            is_glm=False, num_process=args.num_process
        )
    elif args.mode == "response":
        build_training_file(
            input_file=args.input_file,
            output_file=args.input_file.replace(".jsonl", f"_{args.backbone}.jsonl"),
            worker_func=partial(standard_prompt_response, skip_response=args.skip_response, skip_generated=args.skip_generated, backbone=args.backbone, key=args.prompt_key, num_generation=args.num_generation),
            is_glm=False,
            response_key=args.response_key
        )
    else:
        raise NotImplementedError 

