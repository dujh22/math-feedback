# 如果打开下面一行，命令行会自动输出代码执行的时间
import time
# 这个装饰器 time_it 可以被应用到任何你希望测量执行时间的函数上。它通过计算函数开始和结束时的时间来计算执行时间，并将时间转换为小时、分钟和秒的格式。
def time_it(func):
    """
    装饰器，用于测量函数执行时间。
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 获取开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 获取结束时间
        time_taken = end_time - start_time  # 计算耗时
        hours, rem = divmod(time_taken, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"{func.__name__} executed in: {int(hours):02d}h:{int(minutes):02d}m:{seconds:06.3f}s")
        return result
    return wrapper

# 如果打开下面两行，命令行会自动输出代码执行的全部日志
# import hunter # 用于调试
# hunter.trace(module=__name__) # 用于调试

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.config import CRITIC_URL # 从config.py中导入CRITIC_URL
from llm.config import TGI_URL # Import TGI_URL from config.py

# 本脚本对多进程支持，用于处理大数据

import argparse  # 导入argparse模块，用于处理命令行参数
import time  # 导入time模块，用于时间相关的函数
import json  # 导入json模块，用于处理JSON数据格式
import random  # 导入random模块，用于生成随机数
import requests  # 导入requests模块，用于HTTP请求
from tqdm import tqdm  # 从tqdm模块导入tqdm，用于显示进度条
from functools import partial  # 从functools模块导入partial函数，用于固定函数的部分参数
from multiprocess import Pool, Queue, Process  # 从multiprocess模块导入Pool, Queue, Process，用于多进程处理
import warnings  # 导入warnings模块，用于警告控制
import openai  # 导入openai模块，用于调用OpenAI的API
import re  # 导入re模块，用于正则表达式操作

random.seed(666)  # 设置随机数生成器的种子，确保结果具有可重复性
warnings.filterwarnings("ignore")  # 忽略警告信息

NUM_PROCESS=10  # 设置并发处理的进程数量

QUEUE_SIZE=1000000  # 设置队列的最大容量

TEMPERATURE = 0.9  # 设置生成文本时的温度参数
TOPP = 0.2  # 设置生成文本时的Top-p参数
PROMPT_TEMPLATE = None  # 初始化提示模板变量


def query_chatglm_platform(prompt, history=[], do_sample=True, max_tokens=2048, url = CRITIC_URL):
    '''
        该功能用于与基于聊天的模型平台通信，向其发送当前和历史对话数据，然后接收生成的回复。它可以通过温度和top_p 等参数对生成过程进行详细定制，因此适用于需要根据用户输入和以前的对话上下文动态生成回复的交互式应用。
    '''
    # url = "http://xxx:9090/v1/chat/completions"  # 设置API的URL
    url = CRITIC_URL  # 设置API的URL

    messages = []  # 初始化消息列表
    for turn in history:  # 遍历历史记录，每一项都包含用户和助手的对话
        messages.append({
            "role": "user",
            "content": turn["prompt"],  # 用户的提示
        })
        messages.append({
            "role": "assistant",
            "content": turn["response"],  # 助手的回答
        })
    messages.append({  # 添加当前轮次的用户输入
        "role": "user",
        "content": prompt,
    })

    payload = {  # 构造请求载荷
        "messages": messages,  # 包含整个对话历史和当前提示的消息列表
        "temperature": TEMPERATURE,  # 设置温度参数，影响生成文本的随机性
        "top_p": TOPP,  # 设置Top-p参数，控制生成的集中性
        # "model": self.model_version,  # 如果有模型版本控制，可以取消注释此行并指定模型
        "max_tokens": max_tokens,  # 设置最大生成令牌数
        "do_sample": do_sample,  # 是否启用随机采样生成
        "stream": False,  # 是否启用流模式，适用于逐步生成长文本
        "seed": random.randint(1, 10000000),  # 设置随机种子以保证生成的一致性
    }

    # 发送POST请求到服务器并接收响应
    # response = requests.post(self.url, data=payload, headers=self.headers, verify=False)
    response = requests.post(url, json=payload, verify=False)
    
    if response.status_code == 200:  # 如果响应状态码为200，表示成功
        answer = json.loads(response.text)  # 解析响应文本为JSON
        # 下面的注释代码可用于处理特定的结束情况
        # if answer["choices"][0]["finish_reason"] != "eos_token":
            # answer = None
        # else:
        answer = answer["choices"][0]["message"]["content"]  # 获取生成文本
    else:
        print(response.text)  # 打印错误信息
        answer = None  # 出错时返回None

    return answer  # 返回生成的回答或None

def query_chatglm_tgi(prompt, history=[], do_sample=True, max_tokens=2048, max_retry=3, url = TGI_URL):
    '''
        该函数根据对话历史和当前提示构建消息流，然后查询指定 URL 上的文本生成模型。它会调整生成参数，如采样、标记限制和温度，并在出现错误时重试请求。这种功能对于将历史对话上下文整合到响应生成中的系统来说非常典型，因此适用于需要保持连贯和上下文适当的交互的聊天应用或对话系统。
    '''
    # url = "http://xxx:8080/generate"  # 设置API的URL
    # url = TGI_URL 
    messages = ""  # 初始化消息字符串
    for turn in history:
        ques, ans = turn["prompt"], turn["response"]  # 从历史中获取问题和回答
        messages += f"<|user|>\n{ques}<|assistant|>\n{ans}"  # 将问题和回答按格式追加到消息字符串

    messages += f"<|user|>\n{prompt}<|assistant|>\n"  # 将当前的提示追加到消息字符串

    inputs = {  # 创建请求体
        "inputs": messages,  # 包含历史和当前提示的消息文本
        "stream": False,  # 不使用流式传输
        "parameters": {  # 设定生成参数
            "best_of": 1,  # 生成的最佳答案数量
            "decoder_input_details": True,  # 是否返回解码器的输入细节
            "details": False,  # 是否返回详细信息
            "do_sample": do_sample,  # 是否启用随机采样生成
            "max_new_tokens": max_tokens,  # 设置最大新令牌数
            "return_full_text": False,  # 是否返回完整文本
            "seed": None,  # 设置随机种子（None表示不固定）
            "temperature": 1,  # 设置生成文本的温度
            "top_p": 0.9,  # 设置Top-p参数，控制生成的集中性
            "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"]  # 设置停止符号
        }
    }

    for _ in range(max_retry):  # 最多重试指定次数
        output = requests.post(url, json=inputs)  # 发送POST请求
        if output.status_code == 200:  # 如果服务器响应状态码为200
            output = json.loads(output.text)  # 解析响应内容
            # results.append(output[0]["generated_text"])
            result = output["generated_text"]  # 获取生成的文本
            break  # 成功获取则退出循环
        else:
            print(output.text)  # 打印错误信息
    else:  # 如果重试次数耗尽仍未成功
        result = None  # 将结果设置为None

    return result  # 返回生成的结果或None

def query_gpt4(prompt, history=[], backbone="gpt-3.5-turbo"):
    '''
        该函数与 OpenAI 的 API 集成，可根据一系列对话转折和当前用户提示生成回复。它使用 "用户 "和 "助手 "的角色设置对话历史，然后向指定的 OpenAI 模型发出请求以生成回复。该功能通过温度和 top_p 等参数进行配置，以控制生成回复的随机性和集中可能性。这对聊天机器人或对话式代理特别有用，因为在这些情况下，保持上下文和生成连贯的回复至关重要。
    '''
    messages = []  # 初始化消息列表

    # 将历史对话转换成消息列表格式，包括用户和助手的每次交互
    for turn in history:
        messages.append({"role": "user", "content": turn["prompt"]})  # 添加用户的部分
        messages.append({"role": "assistant", "content": turn["response"]})  # 添加助手的部分
    messages.append({"role": "user", "content": prompt})  # 添加当前轮次用户的输入

    # 使用OpenAI API创建聊天完成请求
    # create a chat completion
    chat_completion = openai.ChatCompletion.create(
                            model=backbone,  # 指定使用的模型版本
                            messages=messages,  # 传入构建好的对话历史
                            temperature=TEMPERATURE,  # 设置温度参数，影响生成文本的随机性
                            top_p=TOPP,  # 设置Top-p参数，控制生成的集中性
                            # max_tokens=2048,  # 可以指定最大令牌数，这里被注释掉
                        )
    return chat_completion.choices[0].message.content  # 返回生成的文本内容

def query_gpt4_with_standard_format(inputs):
    '''
        该功能使用 OpenAI 的 GPT-4 模型，根据提供的对话信息生成响应。该功能专为与较新版本的模型（"gpt-4-1106-preview"）进行交互而定制，与早期模型相比，具有更先进的回复生成功能。该功能非常适合需要最先进的自然语言处理功能的应用，如高级聊天机器人、虚拟助手或其他从最新语言模型开发中受益的交互系统。
    '''
    messages = inputs["messages"]  # 从输入中获取消息列表

    # 使用OpenAI的API创建聊天完成请求
    chat_completion = openai.ChatCompletion.create(
                            model="gpt-4-1106-preview",  # 指定使用的模型版本为GPT-4的预览版
                            messages=messages,  # 传入构建好的对话消息
                            temperature=TEMPERATURE,  # 设置温度参数，影响生成文本的随机性
                            top_p=TOPP,  # 设置Top-p参数，控制生成的集中性
                            # max_tokens=2048,  # 可以指定最大令牌数，这里被注释掉
                        )
    return chat_completion.choices[0].message.content  # 返回生成的文本内容

# 定义处理任务的工作函数，是build_training_file函数的子函数
def worker_build_training_pair(task_queue, done_queue, worker_func, max_retry=3, is_glm=False): 
    '''
        该函数是一个工作进程，旨在从任务队列中提取任务，使用指定函数进行处理，然后将处理结果放入已完成队列。它将继续处理任务，直到收到 "STOP"（停止）信号，表明没有更多任务。函数会尝试处理每个任务，最多可重试 max_retry次，并在必要时处理异常和重试。如果一个任务在允许的重试次数后未能成功处理，它就会跳到下一个任务。一旦所有任务都处理完毕，函数就会向已完成队列添加一条 "COMPLETE "消息，以示处理完成。
    '''
    for line in iter(task_queue.get, "STOP"):  # 从任务队列中持续获取任务，直到接收到"STOP"信号
        item = json.loads(line)  # 将从队列中取得的JSON字符串解析为Python字典
        response = None  # 初始化响应变量
        for _ in range(max_retry):  # 尝试最多max_retry次来执行工作函数
            try:
                response = worker_func(item)  # 调用工作函数处理项目
            except Exception as e:  # 捕获并处理可能的异常
                print("error:", e)  # 打印错误信息
                # exit()
                continue  # 忽略错误，继续下一次尝试

            if response is not None:  # 如果成功获得响应
                break  # 退出重试循环
            
            time.sleep(3)  # 如果没有获得有效响应，等待3秒后重试
        else:  # 如果重试次数耗尽仍未获得有效响应，则继续处理下一个项目
            continue

        done_queue.put(item)  # 将处理完成的项目放入完成队列

    done_queue.put("COMPLETE")  # 所有任务处理完成后，放入"COMPLETE"信号到队列

# 定义构建训练文件的函数：多子进程
def build_training_file(input_file, output_file, worker_func, is_glm=False, num_process=None): 
    '''
        本代完成创建多进程设置，以便从输入文件读取数据并使用 Worker 函数进行处理。每个进程都从任务队列中提取任务，对其进行处理，并将结果推送到完成队列。主进程从已完成队列中读取结果并将其写入输出文件，同时用进度条显示进度。
    '''
    if num_process is None:
        num_processes = NUM_PROCESS  # 如果未指定num_process，则使用默认的NUM_PROCESS常量
    else:
        num_processes = num_process  # 否则使用指定的num_process
        
    # 创建任务队列和完成队列，每个队列的最大容量为QUEUE_SIZE
    task_queue, done_queue = Queue(maxsize=QUEUE_SIZE), Queue(maxsize=QUEUE_SIZE)

    def read_data_into_queue():
        cnt = 0  # 初始化计数器
        
        with open(input_file, "r", encoding="utf-8") as r:  # r以只读she模式打开输入文件
            print("read files")
            for line in r:  # 逐行读取文件
                task_queue.put(line)  # 将行数据放入任务队列
                cnt += 1  # 计数器递增
            print("read files done: ", cnt)  # 打印读取完成后的行数

        # 向任务队列中添加STOP信号，每个处理进程一个
        for _ in range(num_processes):
            task_queue.put('STOP')

    processes = []  # 初始化进程列表
    for _ in range(num_processes):  # 为每个进程创建一个子进程
        process = Process(target=partial(worker_build_training_pair, is_glm=is_glm),
                    args=(task_queue, done_queue, worker_func))  # 设置子进程目标函数和参数
        process.start()  # 启动子进程
        processes.append(process)  # 将子进程添加到进程列表

    process = Process(target=read_data_into_queue)  # 创建数据读取子进程
    process.start()  # 启动数据读取子进程

    progress_bar = tqdm()  # 初始化进度条
    print("----- GOGOGOGOGOGOGO !!!!!")
    with open(output_file, 'w', encoding='utf-8') as w:  # 以写入模式打开输出文件
        num_finished = 0  # 初始化已完成进程数
        num_save = 0  # 初始化已保存项目数
        while num_finished < num_processes:  # 循环，直到所有处理进程完成
            item = done_queue.get()  # 从完成队列获取项目
            if item == 'COMPLETE':  # 如果接收到完成信号
                num_finished += 1  # 增加已完成进程数
            else:
                w.write(json.dumps(item, ensure_ascii=False) + '\n')  # 将项目写入文件
                w.flush()  # 刷新文件缓冲区，确保写入磁盘
                num_save += 1  # 更新已保存项目数
                progress_bar.update(1)  # 更新进度条

@time_it
def standard_prompt_response(  
    x, 
    response_key="response", 
    skip_response=False, 
    skip_generated=False, 
    backbone="gpt-3.5-turbo", 
    prompt_key="prompt", 
    num_generation=1,
    url = TGI_URL
): 
    '''
       该函数是一个通用包装器，可根据指定的提示和后端模型生成响应，并可配置生成次数和重试次数。它支持在特定条件下跳过响应，并与包括 GPT 模型变体在内的各种后端模型集成。它的设计目的是处理提供交互历史记录的情况，并在批处理过程中随机打印出响应，用于调试或监控。 
    '''
    # 如果设置了跳过响应且响应键存在于字典中，直接返回该响应
    if skip_response and response_key in x:
        print("skip")
        return x[response_key]
    
    # 如果响应键存在于字典中，直接返回该响应
    if response_key in x: 
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
    
    # 判断是否包含历史对话，如果有则使用，否则为空列表
    if "history" in x:
        history = x["history"]
    else:
        history = []

    # 从字典中获取提示文本
    prompt = x[prompt_key]

    # 初始化响应列表
    responses = []
    for i in range(num_generation):  # 根据所需的生成次数进行循环
        max_try = 3  # 最大尝试次数为3
        for _ in range(max_try):
            if backbone == "chatglm_platform":
                result = query_chatglm_platform(prompt=prompt, history=history, url=url)  # 如果后端是chatglm_platform，调用相应的查询函数
            elif backbone == "tgi":
                result = query_chatglm_tgi(prompt=prompt, history=history, url=url)  # 如果后端是tgi，调用相应的查询函数
            elif backbone == "chatglm_ipo":
                # result = query_chatglm(prompt, history)
                raise NotImplementedError  # 如果后端是chatglm_ipo，抛出未实现异常
            elif "gpt" in backbone:
                result = query_gpt4(prompt, history, backbone=backbone)  # 如果后端是gpt系列，调用相应的查询函数
            else:
                raise NotImplementedError  # 如果后端未知，抛出未实现异常

            if result is None:
                continue  # 如果结果为空，继续尝试
        if result is not None:
            responses.append((f"reply_{i}", result))  # 将有效结果添加到响应列表

    import random

    # 如果生成了结果，随机打印其中的一个
    if len(responses) > 0:        
        rnm = random.randint(0, 20)
        if rnm == 0:
            print(f"#### Question: {prompt} ------ \n Response: ", responses[0][1])        
            print()

    # 如果只需要生成一个响应，取列表中的第一个
    if num_generation == 1:
        result = responses[0][1]

    # 将生成的响应存储在字典x中指定的键下
    x[response_key] = result
    return result
    
@time_it
def critic_math_problem(x, backbone="chatglm_platform", prompt_key="prompt", response_key="response", reference_key="answer", max_retry=3, PROMPT_TEMPLATE=None, url = CRITIC_URL):
    '''
        该函数使用指定的模型（主干）评估数学应答。它用问题陈述、正确答案和助手的回答格式化输入，然后查询模型以评估回答的准确性。该函数会多次尝试以获得评级，并将结果添加到输出列表中，其中包括答案、评级和完整的判断结果。这种设置通常用于需要对回答进行自动评分或反馈的教育或测试环境中。
    '''
    # prompt = 
    # 获取响应数据，如果响应是字符串，则转换为列表形式
    response = x[response_key]
    if isinstance(response, str):
        response = [response]
    prompt = x[prompt_key]  # 获取提示信息

    outputs = []  # 初始化输出列表
    for resp_item in response:  # 遍历响应列表
        rating = None  # 初始化评分变量
        if isinstance(resp_item, str):
            resp = resp_item  # 如果响应项是字符串，直接使用
        elif isinstance(resp_item, list) or isinstance(resp_item, tuple):
            resp = resp_item[-1]  # 如果响应项是列表或元组，使用最后一个元素
        else:
            raise NotImplementedError  # 如果响应项格式不支持，抛出未实现异常

        for _ in range(max_retry):  # 尝试最大重试次数
            input_data = PROMPT_TEMPLATE.format(
                problem=prompt,
                reference_answer=x.get(reference_key, ""),
                assistant_ansiwer=resp
            )  # 格式化输入数据

            # 根据选择的后端模型调用相应的查询函数
            if backbone == "chatglm_platform":
                result = query_chatglm_platform(prompt=input_data, url=url)
            elif backbone == "tgi":
                result = query_chatglm_tgi(prompt=input_data, url=url)
            elif backbone == "chatglm_ipo":
                # result = query_chatglm(input_data)
                raise NotImplementedError
            else:
                raise NotImplementedError

            rating = re.findall(r"\[\[(\d+)\]\]", result)  # 从结果中提取评分

            if len(rating) == 0:
                continue  # 如果没有找到评分，继续重试
            else:
                rating = rating[0]  # 获取评分
                break  # 退出循环
            
        if rating is not None:
            outputs.append({
                "response": resp,
                "rating": rating,
                "judge_result": result
            })  # 将评分结果添加到输出列表

    x["critic_result"] = outputs  # 将输出结果存入输入字典中

    return x

# 准备模板函数
def prepare_template(prompt_filepath): 
    print(f"Load prompt template from {prompt_filepath}...")  # 打印加载提示模板的文件路径信息
    global PROMPT_TEMPLATE  # 声明PROMPT_TEMPLATE为全局变量
    PROMPT_TEMPLATE = open(prompt_filepath, encoding='utf-8').read().strip()
    return PROMPT_TEMPLATE

def main():
    code_test = False # 是否为代码测试
    if code_test == False:
        input_file_path = None
        prompt_template_path = None
        prompt_key = None
        response_key = "response"
        reference_key = "answer"
        backbone = "gpt-3.5-turbo"
        mode = "response"
        url = None
    else:
        prompt_template_path = 'F://code//github//ChatGLM-MathV2//shepherd_prm//templates//criticllm_math_template.txt'
        prompt_key = "question"
        response_key = "response"
        reference_key = "solution"
        # 下面三个参数需要根据mode动态调整
        
        # 如果是生成模式
        # backbone = "tgi" # generate用tgi，critic用chatglm_platform
        # input_file_path = "F://code//github//ChatGLM-MathV2//data//test_data100//test_data100.jsonl"
        # mode = "response"
        # url = TGI_URL

        # 如果是评估模式
        backbone = "chatglm_platform"
        input_file_path = "F://code//github//ChatGLM-MathV2//data//test_data100//test_data100_tgi.jsonl"
        mode = "critic"
        url = CRITIC_URL

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()  

    parser.add_argument("--input_file", type=str, default=input_file_path)  # 添加输入文件路径的命令行参数
    parser.add_argument("--mode", type=str, default=mode)  # 添加模式选择的命令行参数
    parser.add_argument("--backbone", type=str, default=backbone)  # 添加使用的模型版本的命令行参数
    parser.add_argument("--prompt_key", type=str, default=prompt_key)  # 添加提示关键字的命令行参数
    parser.add_argument("--skip_response", action="store_true", default=False)  # 添加是否跳过响应的命令行参数
    parser.add_argument("--skip_generated", action="store_true", default=False)  # 添加是否跳过生成的命令行参数
    parser.add_argument("--prompt_template", type=str, default=prompt_template_path)  # 添加提示模板的命令行参数
    parser.add_argument("--reference_key", type=str, default=reference_key)  # 添加参考答案关键字的命令行参数
    parser.add_argument("--response_key", type=str, default=response_key)  # 添加响应关键字的命令行参数
    parser.add_argument("--num_generation", type=str, default=1)  # 添加生成数量的命令行参数
    parser.add_argument("--num_process", type=int, default=10)  # 添加处理数量的命令行参数
    parser.add_argument("--url", type=str, default=None)  # 添加API服务器的URL的命令行参数

    args = parser.parse_args()  # 解析命令行参数
    
    if args.mode == "critic":  # 如果模式为评估
        PROMPT_TEMPLATE = prepare_template(args.prompt_template)  # 准备提示模板
        # 构建训练文件
        build_training_file(
            input_file=args.input_file,
            output_file=args.input_file.replace(".jsonl", "_math_critic.jsonl"),  # 设置输出文件路径
            worker_func=partial(
                critic_math_problem, 
                backbone=args.backbone, 
                prompt_key=args.prompt_key, 
                reference_key=args.reference_key,
                response_key=args.response_key,
                PROMPT_TEMPLATE = PROMPT_TEMPLATE,
                url=args.url
            ),
            is_glm=False, 
            num_process=args.num_process  # 设置是否使用GLM模型和处理数量
        )

    elif args.mode == "response":  # 如果模式为响应
        # 构建训练文件
        build_training_file(
            input_file=args.input_file,
            output_file=args.input_file.replace(".jsonl", f"_{args.backbone}.jsonl"),  # 设置输出文件路径
            worker_func=partial(
                standard_prompt_response, 
                skip_response=args.skip_response, 
                skip_generated=args.skip_generated, 
                backbone=args.backbone, 
                prompt_key=args.prompt_key, 
                response_key=args.response_key,
                url = args.url,
            ),
            is_glm=False,
            num_process=args.num_process  # 设置是否使用GLM模型和处理数量
        )
    else:
        raise NotImplementedError  # 如果模式未实现，则抛出异常
    

if __name__ == '__main__':
    main()

    # q = [
    #     "你好",
    #     "你是谁",
    #     "写个小作文"
    # ]
    # for question in q:
    #     print("Q:", question)
    #     print("A", query_chatglm_tgi(""))