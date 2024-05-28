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

from tqdm import tqdm
import json
import re  # 导入re模块，用于正则表达式操作
import requests  # 导入requests模块，用于HTTP请求
import random  # 导入random模块，用于生成随机数
import openai  # 导入openai模块，用于调用OpenAI的API
from llm.config import CRITIC_URL, TGI_URL

TEMPERATURE = 0.9  # 设置生成文本时的温度参数
TOPP = 0.2  # 设置生成文本时的Top-p参数

# 设置模板路径
prompt_template_path = 'F://code//github//ChatGLM-MathV2//shepherd_prm//templates//criticllm_math_template.txt'

@time_it
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
def query_chatglm_platform(prompt, history=[], do_sample=True, max_tokens=2048):
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

@time_it
def query_chatglm_tgi(prompt, history=[], do_sample=True, max_tokens=2048, max_retry=3):
    '''
        该函数根据对话历史和当前提示构建消息流，然后查询指定 URL 上的文本生成模型。它会调整生成参数，如采样、标记限制和温度，并在出现错误时重试请求。这种功能对于将历史对话上下文整合到响应生成中的系统来说非常典型，因此适用于需要保持连贯和上下文适当的交互的聊天应用或对话系统。
    '''
    # url = "http://xxx:8080/generate"  # 设置API的URL
    url = TGI_URL 
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

@time_it
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

@time_it
def standard_prompt_response(  
    x, 
    response_key="response", 
    skip_response=False, 
    skip_generated=False, 
    backbone="gpt-3.5-turbo", 
    prompt_key="prompt", 
    num_generation=1
): 
    '''
       该函数是一个通用包装器，可根据指定的提示和后端模型生成响应，并可配置生成次数和重试次数。它支持在特定条件下跳过响应，并与包括 GPT 模型变体在内的各种后端模型集成。它的设计目的是处理提供交互历史记录的情况，并在批处理过程中随机打印出响应，用于调试或监控。 
    '''
    # 如果设置了跳过响应且响应键存在于字典中，直接返回该响应
    if skip_response and response_key in x:
        print("skip")
        return x[response_key]
    
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
                result = query_chatglm_platform(prompt, history)  # 如果后端是chatglm_platform，调用相应的查询函数
            elif backbone == "tgi":
                result = query_chatglm_tgi(prompt, history)  # 如果后端是tgi，调用相应的查询函数
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

    # 如果只需要生成一个响应，取列表中的第一个
    if num_generation == 1:
        result = responses[0][1]

    # 将生成的响应存储在字典x中指定的键下
    x[response_key] = result
    return result

@time_it
def critic_math_problem(x, backbone="chatglm_platform", prompt_key="prompt", response_key="response", reference_key="answer", max_retry=3, PROMPT_TEMPLATE=None):
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
                reference_answer=x[reference_key],
                assistant_ansiwer=resp
            )  # 格式化输入数据

            # 根据选择的后端模型调用相应的查询函数
            if backbone == "chatglm_platform":
                result = query_chatglm_platform(input_data)
            elif backbone == "tgi":
                result = query_chatglm_tgi(input_data)
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
@time_it
def prepare_template(prompt_filepath): 
    global PROMPT_TEMPLATE  # 声明PROMPT_TEMPLATE为全局变量
    PROMPT_TEMPLATE = open(prompt_filepath, encoding='utf-8').read().strip()
    return PROMPT_TEMPLATE

@time_it
def query_tgi_completion(prompt):
    '''
        这段 Python 代码会配置并向文本生成 API 发送 POST 请求，其中的特定参数会影响生成文本的风格和多样性。这些配置的温度和 top_p 值各不相同，会影响文本生成的确定性或创造性。随机模块用于在这些配置之间进行选择，从而在生成过程中引入可变性。这种设置尤其适用于需要可控和多样化文本输出的应用，例如自动内容生成或聊天机器人。
    '''
    # url = "http://xxx:8080/generate"  # 设置API的URL地址
    url = TGI_URL
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

@time_it
def generate_process(x, prompt_key, response_key, num_path=3, backbone="glm-code-v3"):
    '''
        该函数处理包含提示和响应详细信息的给定字典 x，为响应的每个步骤生成扩展路径。它使用辅助函数 query_tgi_completion，尝试生成扩展路径，每一步最多可生成三次，以确保稳健的错误处理和重试机制。这种方法适用于需要根据先前步骤顺序生成内容的场景，例如为机器学习模型或自动应答系统创建训练数据。
    '''

    prompt = x[prompt_key]  # 从字典x中获取提示信息
    response = x[response_key]  # 从字典x中获取响应信息
    output = []  # 初始化输出列表，用来存储所有生成的扩展路径
    steps = split_response(response)  # 使用split_response函数分割响应文本成多个步骤

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
                    result = query_tgi_completion(query_prompt)  # 调用query_tgi_completion函数尝试获取生成结果
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
def evaluate_process(x, prompt_key="prompt", process_response_key="generated_paths", reference_answewr_key="reference", max_retry=3, backbone="chatglm_platform", PROMPT_TEMPLATE=None):
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
                reference_answewr_key: x[reference_answewr_key]  # 提取参考答案信息
            }
            result = critic_math_problem(  # 调用批评函数对每个响应进行评分
                temp_item,
                backbone=backbone,  # 指定使用的模型后端
                prompt_key=prompt_key,
                response_key="response",
                reference_key=reference_answewr_key,
                PROMPT_TEMPLATE=PROMPT_TEMPLATE
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

@time_it
def select_math_data_by_rating(data):
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

@time_it
def out_to_file(data):
    results = [json.dumps(data, ensure_ascii=False) + '\n']
    with open("output.jsonl", 'w', encoding='utf-8') as file:
        file.writelines(results)

@time_it
def api_both(question, response = None, answer = None):
    data = data = {"question": question}
    
    # 如果提供了回答，就用回答作为回答, 否则生成回答
    if response:
        data["response"] = response
    else:
        print("第零步 生成回答……")
        data["response"] = standard_prompt_response(
            data, 
            backbone = "tgi",
            prompt_key = "question",
            response_key = "response"
        )

    # 如果没有提供标准答案
    if answer == None:
        data["answer"] = "no reference answer"

    # 后向结果评分反馈
    print("第一步 后向结果评分反馈……")
    PROMPT_TEMPLATE = prepare_template(prompt_template_path) # 准备提示模板
    data_back = critic_math_problem(
        data, 
        backbone= "chatglm_platform",
        prompt_key = "question",
        reference_key = "answer",
        response_key = "response",
        PROMPT_TEMPLATE = PROMPT_TEMPLATE
    )

    # 前向过程路径预测
    print("第二步 前向过程路径预测……")
    data_path_pred = generate_process(
        data_back,
        prompt_key = "question",
        response_key = "response"
    )
    
    # 前向过程路径评估
    print("第三步 前向过程路径评估……")
    data_path_pred_judge = evaluate_process(
        data_path_pred,
        backbone = "chatglm_platform",
        prompt_key = "question",
        process_response_key = "generated_paths",
        reference_answewr_key = "answer",
        PROMPT_TEMPLATE = PROMPT_TEMPLATE
    )

    print("第四步 选择数学数据……")
    data_path_pred_judge_aggregate = select_math_data_by_rating(
        data_path_pred_judge
    )

    # 如果有到导出文件的必要，可打开下面一行
    out_to_file(data_path_pred_judge_aggregate)
    
    return data_path_pred_judge_aggregate

@time_it
def main():
    question = "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?"
    result = api_both(question)
    result = json.dumps(result, indent=4, ensure_ascii=False)
    print(result)

@time_it
def main2():
    with open("F://code//github//ChatGLM-MathV2//data//test_data//test_data1.jsonl", 'r', encoding='utf-8') as file, open("F://code//github//ChatGLM-MathV2//data//test_data//test_data1_result.jsonl", 'w', encoding='utf-8') as out_file:    
        for line in tqdm(file, desc="Processing"):
            data = json.loads(line)
            question = data["question"]
            result = api_both(question)
            out_file.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main2()