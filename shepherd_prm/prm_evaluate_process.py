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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.config import TGI_URL # Import TGI_URL from config.py
from llm.config import CRITIC_URL

import json  # 导入json模块，用于处理JSON数据
import argparse  # 导入argparse模块，用于处理命令行参数
from functools import partial  # 从functools模块导入partial，用于固定函数的部分参数
import re  # 导入re模块，用于正则表达式操作
import requests  # 导入requests模块，用于进行HTTP请求
import random  # 导入random模块，用于生成随机数

from query_api import build_training_file, critic_math_problem, prepare_template  # 从query_api模块导入三个函数


def query_tgi_completion(prompt, url = TGI_URL):
    '''
        这段 Python 代码会配置并向文本生成 API 发送 POST 请求，其中的特定参数会影响生成文本的风格和多样性。这些配置的温度和 top_p 值各不相同，会影响文本生成的确定性或创造性。随机模块用于在这些配置之间进行选择，从而在生成过程中引入可变性。这种设置尤其适用于需要可控和多样化文本输出的应用，例如自动内容生成或聊天机器人。
    '''
    # url = "http://xxx:8080/generate"  # 设置API的URL地址
    # url = TGI_URL
    configs = [
        {"temperature": 0.1, "top_p": 0.7},  # 配置列表第一个配置项，温度较低，生成内容较为保守
        {"temperature": 0.9, "top_p": 0.9},  # 配置列表第二个配置项，温度较高，生成内容较为多样
    ]
    if random.randint(0, 5) == 0:  # 随机决定使用哪个配置项，有1/6的概率使用第一个配置
        config = configs[0]
    else:
        config = configs[1]  # 默认情况下使用第二个配置

    messages = f"<|user|>\n{prompt}<|assistant|>\n"  # 将当前的提示追加到消息字符串

    inputs = {  # 创建请求体
        "inputs": messages,  # 包含历史和当前提示的消息文本
        "stream": False,  # 不使用流式传输
        "parameters": {  # 设定生成参数
            "best_of": 1,  # 生成的最佳答案数量
            "decoder_input_details": False,  # 是否返回解码器的输入细节
            "details": False,  # 是否返回详细信息
            "do_sample": True,  # 是否启用随机采样生成
            "max_new_tokens": 2048,  # 指定生成的最大令牌数量
            "seed": random.randint(0, 100000),  # 随机种子，用于生成结果的可复现性
            "temperature": config["temperature"],  # 从选定的配置中获取温度参数
            "top_p": config["top_p"],  # 从选定的配置中获取top_p参数，控制生成的集中性
            "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"]  # 设置停止符号
        }
    }

    output = requests.post(url, json=inputs, verify=False)  # 发送POST请求到服务器，携带定义好的数据载荷，verify=False表示不验证SSL证书
    if output.status_code == 200:  # 如果服务器响应状态码为200
        output = json.loads(output.text)  # 解析响应内容
        result = output["generated_text"]  # 获取生成的文本

    return result  # 返回生成的文本

def split_response(response): # 使用正则表达式按换行符分割响应文本
    # 首先判断\n\n存在的数量，如果超过超过一个则按照这个划分
    if response.count('\n\n') >= 2:
        steps = re.split(r"\n\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps
    # 然后判断\n存在的数量，如果超过一个则按照这个划分
    if response.count('\n') >= 2:
        steps = re.split(r"\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps
    # 否则按照句号划分
    else:
        # 使用正则表达式按句号切割非小数点
        if '。' in response:
            steps = re.split(r'。', response)
        else:
            steps = re.split(r'(?<=[^.0-9])\.(?=[^0-9])', response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps

@time_it   
def generate_process(x, prompt_key, response_key, num_path=3, backbone="glm-code-v3", url = TGI_URL):
    '''
        该函数处理包含提示和响应详细信息的给定字典 x，为响应的每个步骤生成扩展路径。它使用辅助函数 query_tgi_completion，尝试生成扩展路径，每一步最多可生成三次，以确保稳健的错误处理和重试机制。这种方法适用于需要根据先前步骤顺序生成内容的场景，例如为机器学习模型或自动应答系统创建训练数据。
    '''

    prompt = x[prompt_key]  # 从字典x中获取提示信息
    response = x[response_key]  # 从字典x中获取响应信息
    output = []  # 初始化输出列表，用来存储所有生成的扩展路径

    if x.get("solution") is None:
        steps = split_response(response)  # 使用split_response函数分割响应文本成多个步骤
    else:
        steps = []
        for step, info in x["solution"].items():
            content = re.sub(r"<<[^>]*>>", "", info["content"]) # 去除<<等式>>
            steps.append(content)

    # 遍历每一个步骤
    for idx in range(len(steps)):
        extension_path = []  # 为当前步骤初始化扩展路径列表
        
        # 为当前步骤生成指定数量的扩展路径
        for _p in range(num_path):
            _step = "\n".join(steps[:idx+1])  # 将当前步骤之前的所有步骤连接成一个新的查询提示
            # query_prompt = f"\n{prompt}\n{_step}"  # 构造新的查询提示，包括原始提示和当前步骤
            # prompt_temp = "请直接开始继续向后生成推理过程，使用换行符分割，不要输出任何无关内容。 \n"
            prompt_temp = "Please start directly to continue the backward generative inference process, use line breaks to split, and don't output any extraneous content. \n"
            query_prompt = f"{prompt}\n{_step}\n\n {prompt_temp}"  # 构造新的查询提示，包括原始提示和当前步骤
            
            result = None  # 初始化结果变量
            for _ in range(3):  # 最多尝试三次获取生成结果
                try:
                    result = query_tgi_completion(prompt=query_prompt, url=url)  # 调用query_tgi_completion函数尝试获取生成结果
                    if result is not None:  # 如果成功获取到结果，终止循环
                        break
                except Exception as e:
                    continue  # 如果在尝试过程中发生异常，忽略异常，继续尝试
            if result is None:
                continue  # 如果三次尝试后仍未获取到结果，跳过当前路径的生成
            
            extension_path.append(result)  # 将获取到的结果添加到扩展路径列表

        output.append({  # 将当前步骤的所有扩展路径添加到输出列表
            "step": _step,
            "extension": extension_path
        })

    x["generated_paths"] = output  # 将生成的所有扩展路径存储在输入字典x中的"generated_paths"键下
    return x  # 返回更新后的字典x

@time_it
def evaluate_process(x, prompt_key="prompt", process_response_key="generated_paths", reference_answer_key="reference", max_retry=3, backbone="chatglm_platform", PROMPT_TEMPLATE=None, url = CRITIC_URL):
    '''
        该功能通过使用批判函数根据参考答案对每个回复步骤进行评分来评估生成文本路径的质量，通常用于评估数学问题或类似内容，其正确性可以客观判断。它根据分数计算软标签和硬标签，其中软标签是二进制结果（分数高于阈值）的平均值，而硬标签则表示大部分分数是否通过了阈值。这种评估在教育软件、自动辅导系统或其他需要对生成的回复进行反馈的应用中特别有用。
    '''
    generated_paths = x[process_response_key]  # 从字典x中获取生成的路径数据

    # 遍历所有生成的路径
    for path in generated_paths:
        step_paths = path["extension"]  # 获取每个路径的扩展步骤列表
        ratings = []  # 初始化评分列表

        # 为每个扩展步骤进行评分
        for step_path in step_paths:
            temp_item = {
                prompt_key: x[prompt_key],  # 提取提示信息
                "response": step_path,  # 提取响应信息
                reference_answer_key: x.get(reference_answer_key, "")  # 提取参考答案信息
            }
            result = critic_math_problem(  # 调用批评函数对每个响应进行评分
                temp_item,
                backbone=backbone,  # 指定使用的模型后端
                prompt_key=prompt_key,
                response_key="response",
                reference_key=reference_answer_key,
                PROMPT_TEMPLATE=PROMPT_TEMPLATE,
                url=url,
            )
            rating = result["critic_result"][0]["rating"]  # 从结果中获取评分
            ratings.append(rating)  # 将评分添加到评分列表

        ratings = [float(item) for item in ratings] # 将字符串转为数字
        path["ratings"] = ratings  # 将评分列表存储在路径信息中
        ratings_binary = [1 if x >= 8 else 0 for x in ratings]  # 将评分转换为二进制标签（8分以上为1，否则为0）
        path["soft_label"] = sum(ratings_binary) / len(ratings_binary)  # 计算软标签，即平均值
        path["hard_label"] = 1 if path["soft_label"] >= 0.5 else 0  # 根据软标签计算硬标签，平均值大于等于0.5则为1，否则为0

    x[process_response_key] = generated_paths  # 将更新后的生成路径存回字典x

    return x  # 返回更新后的字典x

def select_math_data_by_rating(input_file, output_file):
    '''
        该函数处理一组数学问题（或其他类似的评价任务），根据评分选择或过滤这些问题。它支持以文件路径或直接以数据形式输入。该函数根据给定的下限计算每个项目的平均分数和通过率，并用计算出的分数和选择指标更新每个项目。在需要自动评分和反馈以评估和改进学习材料或算法的教育或测试环境中，这一过程尤其有用。
    '''
    if isinstance(input_file, str):  # 检查输入是否为字符串类型的文件路径
        data = [json.loads(x) for x in open(input_file, 'r', encoding='utf-8')]  
        # 从文件读取每行并解析为JSON对象
    else:
        data = input_file  # 如果不是字符串，假设输入已经是数据列表

    def judge_scores(scores, lower_bound=7):
        avg_score = sum(scores) / len(scores)  # 计算所有评分的平均值
        above_bound = [1 if x >= lower_bound else 0 for x in scores]  # 生成一个列表，评分高于阈值的为1，否则为0
        return avg_score, sum(above_bound) / len(above_bound)  # 返回平均分和超过阈值的比例
    
    def func(x, lower_bound=8):

        # results = x["critic_result"]  # 从每个样本中获取评价结果
        # if len(results) == 0:
        #     return None  # 如果没有评价结果，则返回None
        # ratings = [item["rating"] for item in results if isinstance(item["rating"], str)]  # 提取所有评分，确保评分是字符串格式
        # ratings = [float(x) for x in ratings]  # 将所有评分转换为浮点数

         # 从每个样本中获取评价结果
        ratings = []
        for item1 in x['generated_paths']:
            for item2 in item1['ratings']:
                ratings.append(item2)
        if len(ratings) == 0:
            return None # 如果没有评价结果，则返回None

        avg_score, pass_rate = judge_scores(ratings, lower_bound=lower_bound)  # 调用judge_scores计算平均分和通过率
        x["critic_scores"] = {  # 将计算结果存回样本中
            "ratings": ratings,
            "avg_score": avg_score,
            "pass_rate": pass_rate
        }
        return x
    
    processed = [func(x) for x in data if x is not None]  # 处理数据集中的每个样本，并排除None值

    with open(output_file, 'w', encoding='utf-8') as w:  # 以写入模式打开输出文件
        for item in processed:  # 遍历处理后的数据列表
            w.write(json.dumps(item, ensure_ascii=False) + '\n')  # 将每个样本写入文件，并添加换行符

    return processed  # 返回处理后的数据列表

def select_math_data_by_rating2(data):
    '''
        该函数处理一组数学问题（或其他类似的评价任务），根据评分选择或过滤这些问题。它支持以文件路径或直接以数据形式输入。该函数根据给定的下限计算每个项目的平均分数和通过率，并用计算出的分数和选择指标更新每个项目。在需要自动评分和反馈以评估和改进学习材料或算法的教育或测试环境中，这一过程尤其有用。
    '''
    def judge_scores(scores, lower_bound=7):
        avg_score = sum(scores) / len(scores)  # 计算所有评分的平均值
        above_bound = [1 if x >= lower_bound else 0 for x in scores]  # 生成一个列表，评分高于阈值的为1，否则为0
        return avg_score, sum(above_bound) / len(above_bound)  # 返回平均分和超过阈值的比例
    
    def func(x, lower_bound=8):
        # 从每个样本中获取评价结果
        ratings = []
        for item1 in x['generated_paths']:
            for item2 in item1['ratings']:
                ratings.append(item2)
        if len(ratings) == 0:
            return None # 如果没有评价结果，则返回None

        avg_score, pass_rate = judge_scores(ratings, lower_bound=lower_bound)  # 调用judge_scores计算平均分和通过率
        x["critic_scores"] = {  # 将计算结果存回样本中
            "ratings": ratings,
            "avg_score": avg_score,
            "pass_rate": pass_rate
        }
        return x
    
    x = func(data) # 处理数据集中的每个样本，并排除None值

    return x  # 返回处理后的数据列表

def main():
    code_test = False # 是否为代码测试
    if code_test == False:
        prompt_template_path = None
        input_file_path = None
        prompt_key = None
        response_key = "generation"
        reference_key = "answer"
        backbone = "gpt-3.5-turbo"
        mode = "response"
        process_response_key = "generated_paths"
        reference_answer_key = "reference"
        url = None
    else:
        prompt_template_path = 'F://code//github//ChatGLM-MathV2//shepherd_prm//templates//criticllm_math_template.txt'
        prompt_key = "question"
        response_key = "response"
        reference_key = "solution"
        process_response_key = "generated_paths"
        reference_answer_key = "solution"
        # 下面三个参数需要根据mode动态调整

        # 如果是生成模式
        # backbone = "tgi" # generate用tgi，critic用chatglm_platform
        # input_file_path = "F://code//github//ChatGLM-MathV2//data//test_data100//test_data100_tgi_math_critic.jsonl"
        # mode = "generation"
        # url = TGI_URL

        # 如果是评估模式
        backbone = "chatglm_platform"
        input_file_path = "F://code//github//ChatGLM-MathV2//data//test_data100//test_data100_tgi_math_critic_path.jsonl"
        mode = "critic"
        url = CRITIC_URL


    # 创建命令行解析器
    parser = argparse.ArgumentParser()

    # 添加各种命令行参数
    parser.add_argument("--input_file", type=str, default=input_file_path)  # 指定输入文件路径
    parser.add_argument("--mode", type=str, default=mode)  # 指定运行模式
    parser.add_argument("--backbone", type=str, default=backbone)  # 指定使用的模型
    parser.add_argument("--prompt_key", type=str, default=prompt_key)  # 指定提示键名
    # "gpt-4-1106-preview"  # 注释中提及可能使用的模型版本
    parser.add_argument("--skip_response", action="store_true", default=False)  # 是否跳过响应生成
    parser.add_argument("--skip_generated", action="store_true", default=False)  # 是否跳过已生成的响应
    parser.add_argument("--prompt_template", type=str, default=prompt_template_path)  # 添加提示模板的命令行参数
    parser.add_argument("--reference_key", type=str, default=reference_key)  # 指定参考答案键名
    parser.add_argument("--reference_answer_key", type=str, default=reference_answer_key)  # 指定参考答案键名
    parser.add_argument("--response_key", type=str, default=response_key)  # 指定响应键名
    parser.add_argument("--process_response_key", type=str, default=process_response_key)  # 指定响应键名
    parser.add_argument("--num_process", type=int, default=10)  # 添加处理数量的命令行参数
    parser.add_argument("--url", type=str, default=url)  # 指定API的URL地址

    args = parser.parse_args()  # 解析命令行参数
    
    # 根据指定的模式进行相应的操作
    if args.mode == "generation":
        build_training_file(
            input_file=args.input_file,  # 输入文件
            output_file=args.input_file.replace(".jsonl", "_path.jsonl"),  # 输出文件路径
            worker_func=partial(
                generate_process,  # 指定工作函数
                prompt_key=args.prompt_key,  # 传递prompt_key参数
                response_key=args.response_key,  # 传递response_key参数
                url=args.url  # 传递url参数
            ),
            is_glm=False,  # 指定是否使用GLM模型
            num_process=args.num_process  # 传递处理数量参数
        )
    elif args.mode == "critic":
        PROMPT_TEMPLATE = prepare_template(args.prompt_template)  # 准备提示模板

        build_training_file(
            input_file=args.input_file,  # 输入文件
            output_file=args.input_file.replace(".jsonl", "_math_critic.jsonl"),  # 输出文件路径
            worker_func=partial(
                evaluate_process,  # 指定工作函数
                backbone=args.backbone,  # 传递模型参数
                prompt_key=args.prompt_key,  # 传递prompt_key参数
                process_response_key=args.process_response_key,  # 传递response_key参数
                reference_answer_key=args.reference_answer_key, # 传递reference_key参数
                PROMPT_TEMPLATE = PROMPT_TEMPLATE,
                url=args.url  # 传递url参数
            ),  
            is_glm=False,  # 指定是否使用GLM模型
            num_process=args.num_process  # 传递处理数量参数
        )

        select_math_data_by_rating(
            input_file=args.input_file.replace(".jsonl", "_math_critic.jsonl"),
            output_file=args.input_file.replace(".jsonl", "_math_critic2.jsonl")
        )  # 选择评分数据


if __name__ == "__main__":
    main()