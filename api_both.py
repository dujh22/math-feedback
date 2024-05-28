# 如果打开下面两行，命令行会自动输出代码执行的全部日志,如果不需要，可以注释掉
import hunter # 用于调试
hunter.trace(module=__name__) # 用于调试

import json
import re  # 导入re模块，用于正则表达式操作
import requests  # 导入requests模块，用于HTTP请求
import random  # 导入random模块，用于生成随机数
import openai  # 导入openai模块，用于调用OpenAI的API
import sympy as sp
import subprocess

from llm.config import CRITIC_URL, TGI_URL

TEMPERATURE = 0.9  # 设置生成文本时的温度参数
TOPP = 0.2  # 设置生成文本时的Top-p参数


# 该部分用于设置调用的LLM相关信息
import llm.config as config
# 设定API密钥和基本URL
openai.api_key = config.GPT_API_KEY
openai.api_base = config.GPT_BASE_URL
from chatglm import ChatGLM
ChatGLM = ChatGLM()
USE_GLM_OR_GPT = 'glm'
# 这里设置使用的llm进行生成，注意在本项目中只有这里一个地方进行相关设置
def llm_response(prompt, use_glm_or_gpt = USE_GLM_OR_GPT):
    response = "ERROR for LLM"
    for i in range(10):
        if use_glm_or_gpt == 'glm':
            try:
                response = ChatGLM.generate(prompt)
                return response
            except:
                continue
        else:
            try:
                # 构造messages
                messages = [{"role": "user", "content": prompt}]
                # 调用GPT接口
                # model = "gpt-3.5-turbo"
                model = "gpt-4-1106-preview"
                chat_completion = openai.ChatCompletion.create(model=model, messages = messages)
                response = chat_completion.choices[0].message.content
                return response
            except:
                continue
    return response

# 设置模板路径
prompt_template_path = 'F://code//github//ChatGLM-MathV2//shepherd_prm//templates//criticllm_math_template.txt'

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
def prepare_template(prompt_filepath): 
    global PROMPT_TEMPLATE  # 声明PROMPT_TEMPLATE为全局变量
    PROMPT_TEMPLATE = open(prompt_filepath, encoding='utf-8').read().strip()
    return PROMPT_TEMPLATE

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

def out_to_file(data):
    results = [json.dumps(data, ensure_ascii=False) + '\n']
    with open("output.jsonl", 'w', encoding='utf-8') as file:
        file.writelines(results)

def run_python(code, timeout=None):
    result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=timeout)
    stdout = result.stdout
    stderr = result.stderr
    return stdout, stderr

def all_alpha(str):
    if str == 'x': # 特例
        return False
    if all(temp.isalpha() for temp in str):
        return True
    else:
        return False

def highlight_equations(text):
    parts2 = text.split(' ')
    if len(parts2) == 1:
        return text  # 没有等号，返回原文本
    
    # 预处理，如果数字后面紧跟着一个字母串，可以删去这个字母串
    parts = []
    i = 0
    while i < len(parts2):
        parts.append(parts2[i])
        if parts2[i].isdigit():
            if i + 1 < len(parts2) and all_alpha(parts2[i + 1]):
                i += 2
            else:
                i += 1
        else:
            i += 1

    result = []
    last_pos = 0
    
    for i in range(len(parts)):
        # 找到=所在的位置
        if '=' not in parts[i]:
            continue
        else:
            start_pos = i
            end_pos = i
            if start_pos > 0:
                while all_alpha(parts[start_pos - 1]) == False:
                    start_pos -= 1
                    if start_pos == 0:
                        break
            if end_pos < len(parts) - 1:
                while all_alpha(parts[end_pos + 1]) == False:
                    end_pos += 1
                    if end_pos == len(parts) - 1:
                        break
            # print(parts[start_pos:end_pos + 1])
            equation = ' '.join(parts[start_pos:end_pos + 1])
            equation = equation.replace('=', f'=<<{equation}>>')
            for item in range(last_pos, start_pos):
                result.append(parts[item])
            result.append(equation)
            last_pos = end_pos + 1
    
    # 添加最后一个没有等号的部分
    for item in range(last_pos, len(parts)):
        result.append(parts[item])

    return ' '.join(result)

def SplitByRow(data):
    # 初始化新的JSON格式
    new_json = {
        "question": data["question"],
        "solution": {},
    }

    # 切分成独立的步骤
    solutions = split_response(data['solution'])

    # 处理每个解决方案部分
    for i, solution in enumerate(solutions):
        if solution.strip():  # 确保切割后的文本不为空
            new_json["solution"][f"Step {i+1}"] = {
                "content": solution.strip(),
                "label": 1  # 默认标签为1
            }

    # 处理每个解决方案部分的数学公式高亮
    for step, info in new_json["solution"].items():
        temp_content = info["content"]
        info["content"] = highlight_equations(temp_content)
    
    # 返回新的JSON格式
    return new_json

def is_calculation(content):
    # 检查是否存在常见的计算符号
    if re.search(r'[\+\-\*/=%^]', content):
        return 1
    # 检查括号内是否存在计算符号，需要使用更复杂的正则表达式
    if re.search(r'\([^)]*[\+\-\*/=%^][^)]*\)', content):
        return 1
    return 0

def IsCalculationOrReasoning(data):
    # 遍历每一步的解决方案
    if 'solution' in data:
        for step, info in data['solution'].items(): # step变量会接收步骤的名称（如"Step 1"），而info变量会接收与这个步骤名称对应的字典值。
            # 判断并添加新键
            info['is_calculation_or_reasoning'] = is_calculation(info['content'])
    # 将修改后的数据写回新的JSONL文件
    return data

# 修正函数
# 1.在 content 字符串中查找被 << 和 >> 包围的表达式。
# 2.替换 = 后面直到 >> 的内容为 StepCalculatedCorrectlyResult 的值。
# 3.如果 >> 后的内容（如果存在）与 = 和 >> 之间的内容相同，则也将其替换为 StepCalculatedCorrectlyResult。
# content = "The equation a = 2 results in a = 2, and then we see b = 3, which leads to b = 3."
# equations = ["a = 2", "b = 3"]
# judge = [0, 1]  # 只替换a的结果
# result = ["5", "4"]  # 将a的结果替换为5，b的结果不变（虽然这里b不需要替换）
def replace_calculated_result(content, equations, equations_need, equations_correct_format, judge, result):
    # 这里会有两种情况
    # 情况1：result[i]是正确的数值，那么直接进行错误数值的替换是没有问题的
    # 情况2：result[i]是一个字符串，那么就不是直接替换，而是作为一个补充信息只能。因为不可能把所有的错误数值替换为一个相同的文字串
    for i, equation in enumerate(equations):
        if judge[i] == 0:  # 需要修改的等式            
            # 等式本身是正确的了
            # 分解等式，获取左侧变量和原始结果
            variable, original_result = equation.split('=')
            variable = variable.strip()
            original_result = original_result.strip()
            
            # 构造用于搜索和替换的正则表达式
            search_pattern = re.escape(variable) + r'\s*=\s*' + re.escape(original_result)
            replace_pattern = f'{variable} = {result[i]}'
            
            # 替换等式
            content = re.sub(search_pattern, replace_pattern, content)
            
            # 替换全文中的原结果
            content = re.sub(r'\b' + re.escape(original_result) + r'\b', result[i], content)

            # 等式本身是不正确的，还需要修改等式本身
            if equations_need[i] == 0:
                if equations_correct_format[i] != "ERROR: LLM cannot correct the formula":
                    content = content.replace(variable, str(equations_correct_format[i]))
                else:
                    temp_variable = variable + "(The formula may be wrong)"
                    content = content.replace(variable, temp_variable)
                
    return content


# 删除公式中可能存在的单位
def simplify_expression(expr):
    import re
    result = ""
    i = 0
    while i < len(expr):
        if expr[i].isdigit() or expr[i] in "+-*/().":
            # 如果是数字或者符号，则添加到结果中
            result += expr[i]
            i += 1
        elif expr[i].isalpha():
            # 如果是字母，则向后扫描直到遇到非字母
            start = i
            while i < len(expr) and expr[i].isalpha():
                i += 1
            if i == len(expr):
                # 如果到达字符串末尾，则停止
                break
            elif expr[i] in "+-*/().":
                # 如果字母后是符号，检查后续内容
                if i + 1 < len(expr) and expr[i + 1].isalpha():
                    # 如果符号后是字母，忽略之前的所有字母
                    i += 1
            elif expr[i].isdigit():
                # 如果字母后是数字，忽略之前的所有字母
                i += 1
                continue
        else:
            # 其他情况，直接增加索引
            i += 1

    # 清理可能出现的连续符号错误，例如多个连续的乘除号
    result = re.sub(r'\*+/', '*', result)
    result = re.sub(r'/\*+', '/', result)

    return result

# 公式预处理
def formula_preprocessing(input_str):
    # 删除等号与等号右侧的内容
    input_str = re.sub(r'=.+', '', input_str)

    # 删除货币符号
    input_str = input_str.replace('$', '')

    # 去除数字前的前导零
    input_str = re.sub(r'\b0+(\d+)', r'\1', input_str)
    
    # 处理时间表示 (假设时间表示为小时和分钟，如 4:30 转换为 4/30)
    input_str = re.sub(r'(\d+):(\d+)', r'\1/\2', input_str)

    # 处理百分比
    input_str = re.sub(r'(\d+)%', r'(\1/100)', input_str)

    # 处理分数表示中的空格（假设意图是将 3 3/4 写成 33/4）
    input_str = re.sub(r'(\d+)\s+(\d+)/(\d+)', r'\1\2/\3', input_str)

    # 使用正则表达式添加必要的乘法符号
    input_str = re.sub(r'(\d)(\()', r'\1*\2', input_str)  # 处理数字后跟左括号的情况16+3(16)+7
    input_str = re.sub(r'(\))(\d)', r'\1*\2', input_str)  # 处理右括号后跟数字的情况(1/2)20
    input_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', input_str)  # 处理数字后跟变量的情况2x
    input_str = re.sub(r'([a-zA-Z])(\()', r'\1*\2', input_str)  # 处理变量后跟左括号的情况x(5+2)

    # 处理货币符号
    currency_match = re.match(r'\$(\d+(\.\d+)?)(\*\d+(\.\d+)?)?', input_str)
    if currency_match:
        # 提取金额和可能的乘数
        amount = float(currency_match.group(1))
        multiplier = currency_match.group(3)
        if multiplier:
            # 计算乘积
            result = amount * float(multiplier.strip('*'))
            input_str = f"${result:.2f}"
        input_str = f"${amount:.2f}"
        
    # 删除那些前后没有运算符且前面有数字的字符串，例如 '45 months' -> '45'
    input_str = re.sub(r'(\d+)\s+([a-zA-Z\' ]+)(?![\*\/\-\+\(\)0-9])', r'\1', input_str)
    
    # 处理33.333...3333333333为33.333
    # 寻找省略号的位置
    ellipsis_index = input_str.find('...')
    if ellipsis_index != -1:
        # 找到省略号后面紧跟的数字部分并截断
        end_index = ellipsis_index + 3
        while end_index < len(input_str) and input_str[end_index].isdigit():
            end_index += 1
        
        # 生成新的表达式，省略后面的数字
        input_str = input_str[:ellipsis_index] + input_str[end_index:]
    
    # 重新处理可能由于删除字符串导致的多余空格
    input_str = ' '.join(input_str.split())

    # 删除包含逗号的数字
    input_str = ' '.join([part for part in input_str.split() if ',' not in part])

    # 重新处理可能由于删除字符串导致的多余空格
    input_str = re.sub(r'\s+', ' ', input_str).strip()

    # 删除可能存在的单位
    input_str = simplify_expression(input_str)

    return input_str

def extract_glm_code(text):
    # 使用正则表达式提取代码块
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""  # 如果没有找到匹配项，返回空字符串

def get_llm_calculate_result(input_str, question, content, history):
    # 构建给语言模型的提示
    # prompt = f"""我正在尝试解决一个数学问题，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在计算的步骤是：{content}。\n\n 我需要计算一个数学表达式，这个表达式是：{input_str}。请帮我生成可执行的Python代码用于计算这个表达式，注意代码的最终打印输出应该是这个表达式的值或者化简后的表达式，不需要输出任何提示信息或者单位"""
    prompt = f"""I am attempting to solve a math problem with the following specific question:{question} \n\n The steps I am currently using to solve the problem are: {history}. \n\n The step I am calculating now is: {content}. \n\n I need to compute a math expression which is: {input_str}. Please help me generate executable Python code for calculating this expression, noting that the final printout of the code should be the value of the expression or the simplified expression, without outputting any hints or units."""
    response = ""
    for i in range(10):  # 尝试最多10次以获取有效的代码响应
        # 使用ChatGLM模型生成代码
        response = llm_response(prompt)
        # print("response:", response)
        code = extract_glm_code(response) # 假设响应即为代码
        stdout, stderr = run_python(code)
        if not stderr:  # 如果没有错误，返回结果
            return stdout, code
    return "ERROR: Python scripts not running", response  # 经过10次尝试失败后返回错误信息

def judge_equation_need(expr, question, input_str, history):
    # 构建给语言模型的提示
    # prompt = f"""我正在尝试解决一个数学问题，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在计算的步骤是：{input_str}。\n\n 我需要计算一个数学表达式，这个表达式是：{expr}。请问这个表达式是否正确？\n\n 如果正确请只回答yes。 \n\n 如果不正确，请帮我修正这个表达式，注意表达式应该是可计算的表达式，不能包含任何单位。请返回唯一的正确的表达式，用<<>>包围起来。也就是说，<<>>只出现一次"""
    prompt = f"""I am attempting to solve a math problem with the following specific question:{question} \n\n The steps I am currently using to solve the problem are: {history}. \n\n The step I am currently calculating is: {input_str}. \n\n I need to calculate a math expression which is: {expr}. Is this expression correct? \n\n If it is correct please answer yes only. \n\n If it is not correct, please help me to correct this expression, noting that the expression should be a computable expression and should not contain any units. Please return the only correct expression, surrounded by <<>>. That is, <<>> appears only once."""
    response = ""
    for i in range(10):  # 尝试最多10次以获取有效的代码响应
        # 使用ChatGLM模型生成结果
        response = llm_response(prompt)
        # 获取response的第一句话或者如果没有符号就是完整的response
        response_first_sentence = response.split(".")[0]
        if response_first_sentence[:3].lower() == "yes" or "yes" in response_first_sentence.lower():  # 如果是一个正确的计算公式
            return 1, expr, response
        else:
            # 尝试修正表达式
            match = re.search(r'<<(.+?)>>', response)
            if match:
                temp_expr = match.group(1)
                if temp_expr.strip() != "":
                    return 0, temp_expr, response
    return 0, "ERROR: LLM cannot correct the formula", response  # 经过10次尝试失败后返回错误信息

# 删除小数后面多余的0
def evaluate_and_simplify(expr):
    # 计算表达式的数值结果
    numerical_result = expr.evalf()
    
    # 检查结果是否非常接近某个整数
    if numerical_result.is_Integer:
        return int(numerical_result)  # 结果是整数
    else:
        # 尝试四舍五入看看是否能得到整数
        rounded_result = round(numerical_result)
        if abs(numerical_result - rounded_result) < 1e-10:
            return int(rounded_result)  # 四舍五入后的整数
        else:
            return float(numerical_result)  # 保留为浮点数
        
def get_sympy_calculate_result(input_str, question, content, history):
    use_sympy_or_llm = 'sympy'
    temp_key = input_str
    intermediate_process = input_str
    input_str = formula_preprocessing(input_str) # 公式预处理，采用规则库
    if input_str != intermediate_process:
        intermediate_process = intermediate_process + "=>" + input_str
    code = ""

    # 定义可能的符号变量，确保它们被识别为符号而不是字符串
    symbols = re.findall(r'[a-zA-Z]+', input_str)
    symbols = set(symbols)  # 去除重复项
    local_dict = {s: sp.symbols(s) for s in symbols}

    try:
        # 将字符串转换为sympy表达式 expr = sp.sympify(input_str)
        # 使用 locals 参数显式地将这些变量定义为符号
        expr = sp.sympify(input_str, locals=local_dict)
        if str(expr) != input_str:
            intermediate_process = intermediate_process + "=>" + str(expr)
        # 计算表达式
        result = expr
        if str(result) != str(expr):
            intermediate_process = intermediate_process + "=>" + str(result)
        # 化简表达式
        simplified_expr = sp.simplify(result)
        if str(simplified_expr) != str(result):
            intermediate_process = intermediate_process + "=>" + str(simplified_expr)
        # 检查结果是否为布尔值
        if isinstance(simplified_expr, bool):  # 直接使用 Python 的内建类型 bool
            return simplified_expr, use_sympy_or_llm, intermediate_process, code
        try:
            # 如果是数学表达式，返回计算结果
            actual_result = simplified_expr.evalf()
            # 删除后面可能存在的多余的0
            actual_result = evaluate_and_simplify(actual_result)
            if str(actual_result) != str(simplified_expr):
                intermediate_process = intermediate_process + "=>" + str(actual_result)
            return actual_result, use_sympy_or_llm, intermediate_process, code
        except Exception as e:
            # actual_result = simplified_expr
            use_sympy_or_llm = 'sympy and llm'
            actual_result_temp, code = get_llm_calculate_result(simplified_expr, question, content, history)
            actual_result = formula_preprocessing(actual_result_temp) # 结果预处理，采用规则库
            if actual_result != actual_result_temp:
                intermediate_process = intermediate_process + "\n\n Code simplifued:" + actual_result_temp + "=>" + actual_result
            return actual_result, use_sympy_or_llm, intermediate_process, code
    except Exception as e:
        simplified_expr = input_str
        # actual_result = simplified_expr
        use_sympy_or_llm = 'sympy and llm'
        actual_result_temp, code = get_llm_calculate_result(simplified_expr, question, content, history)
        actual_result = formula_preprocessing(actual_result_temp) # 结果预处理，采用规则库
        if actual_result != actual_result_temp:
            intermediate_process = intermediate_process + "\n\n Code simplifued:" + actual_result_temp + "=>" + actual_result
     
    return actual_result, use_sympy_or_llm, intermediate_process, code

def check_calculation(info, question, history):
    input_str = info['content']
    info['equation'] = []
    info['leftSideOfEqualSign'] = []
    info['rightSideOfEqualSign'] = []
    info['leftSideOfEqual_use_sympy_or_llm'] = []
    info['rightSideOfEqual_use_sympy_or_llm'] = []
    info['leftSideOfEqual_code'] = []
    info['rightSideOfEqual_code'] = []
    info['JudgmentStepCalculatedCorrectly'] = []
    info['StepCalculatedCorrectlyResult'] = []
    info['JudgmentStepEquationCorrectly'] = []
    info['StepEquationCorrectlyFormat'] = []
    info['StepEquationCorrectlyFormatLLMResponse'] = []

    # 使用正则表达式查找计算表达式和结果
    pattern = r"<<(.+?)=(.+?)>>" # <>是必须的, =是必须的
    # pattern = r"<<?(.+?)=(.+?)>>?" # <>是可选的
    matches = re.findall(pattern, input_str)

    # 如果不存在，则需要自行寻找=号，以及其前后的数学表达式
    if len(matches) == 0:
       # 使用正则表达式查找计算表达式和结果
       pattern = r"([\d\s\/*\-+.%]+)=([\d\s\/*\-+.%]+)"
       matches = re.findall(pattern, input_str)

    # 遍历所有匹配项进行检查
    for expr, expected_result in matches: # expr变量会接收表达式的内容（如"20*40"），而expected_result变量会接收表达式的结果（如"800"）。
        # logging.info(f"expr: {expr}, expected_result: {expected_result}")
        # 为什么能根据=号分割，因为=号是必须的
        # 去除头尾的 <<
        expr = expr.lstrip("<")
        # 如果还存在=，保留=前面的，如果不存在=，那就是其本身
        expr = expr.split("=")[0]
        # 去除头尾的 >>
        expected_result = expected_result.rstrip(">")
        # 如果还存在-，则保留=后面的
        expected_result = expected_result.split("=")[-1]

        # 添加表达式
        info['equation'].append(f"{expr}={expected_result}")

        # 判断该表达式所进行的计算是否合适
        judge_eqation_correct, correct_expr, llm_response = judge_equation_need(expr, question, input_str, history) 

        info['JudgmentStepEquationCorrectly'].append(judge_eqation_correct)
        info['StepEquationCorrectlyFormat'].append(correct_expr)
        info['StepEquationCorrectlyFormatLLMResponse'].append(llm_response)
        
        # 判断该表达式是否正确计算
        if judge_eqation_correct == 1: # 如果是正确的计算表达式
            # 使用 sympy 计算表达式的结果
            actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(expr, question, input_str, history)
            expected_result, use_sympy_or_llm2, intermediate_process2, code2 = get_sympy_calculate_result(expected_result, question, input_str, history)
        
            info['leftSideOfEqualSign'].append(intermediate_process1)
            info['rightSideOfEqualSign'].append(intermediate_process2)
            info['leftSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm1)
            info['rightSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm2)
            info['leftSideOfEqual_code'].append(code1)
            info['rightSideOfEqual_code'].append(code2)

            # 比较实际结果和期望结果
            if actual_result != expected_result: # sympify(expected_result).evalf()是将expected_result转换为sympy对象并计算其值，evalf()方法返回计算结果。
                info['JudgmentStepCalculatedCorrectly'].append(0)
                info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")    
            else:
                info['JudgmentStepCalculatedCorrectly'].append(1)
                info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")
        else: # 如果不是正确的计算表达式
            info['JudgmentStepCalculatedCorrectly'].append(0)
            # 如果返回了正确的公式没那么可以进行计算
            if correct_expr != "ERROR: LLM cannot correct the formula":
                # 使用 sympy 计算表达式的结果
                actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(correct_expr, question, input_str, history)
            else:
                actual_result, use_sympy_or_llm1, intermediate_process1, code1 = "ERROR: LLM cannot correct the formula", "sympy and llm", "ERROR: LLM cannot correct the formula", "ERROR: LLM cannot correct the formula"

            info['leftSideOfEqualSign'].append(intermediate_process1) # 如果不是正确的计算表达式，那么左边和右边的表达式都是空的,这两个位置是记录公式推导过程的
            info['rightSideOfEqualSign'].append("")
            info['leftSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm1)
            info['rightSideOfEqual_use_sympy_or_llm'].append("")
            info['leftSideOfEqual_code'].append(code1)
            info['rightSideOfEqual_code'].append("")

            info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")

def JudgmentStepCalculatedCorrectly(data):
    question = data['question']
    history = ""
    history_json = {}
    if 'solution' in data:
        for step, info in data['solution'].items():
            check_calculation(info, question, history)
            temp_content = info['content']
            # 为了方便后续进行本步骤的判断，需要修正历史步骤中的错误部分
            if len(info["JudgmentStepCalculatedCorrectly"]) > 0:
                # 找到info["content"]中错误的部分进行替换
                # 主要是在字符串中找到<<80*2=1600>>1600比如，然后替换1600>>1600
                temp_content = replace_calculated_result(info["content"], info["equation"], info['JudgmentStepEquationCorrectly'], info['StepEquationCorrectlyFormat'], info["JudgmentStepCalculatedCorrectly"], info["StepCalculatedCorrectlyResult"])
            history += f"{step}: {temp_content}\n"
            history_json[step] = temp_content
            # 创建一个history_json的副本，并将其赋值给info['history_json']
            info['history_json'] = history_json.copy()
    return data

# 判断单步推理是否正确
def check_reasoning(per_step, content, question, history):
    # promt = f"""我正在尝试解决一个数学问题，具体问题是：“{question}”。\n\n 我之前的解题步骤如下：“{history}” \n\n 现在我正在推理这一步是：“{per_step}”，具体推理内容是：“{content}”。\n\n 请评估我这一步的推理是否正确。\n\n 如果我目前的推理步骤正确并且与问题相关，只需要回答“yes”。\n\n 如果这一步推理错误或者不相关，需要你进行修正，并直接提供正确或更相关的推理步骤，用<<>>包围起来。"""
    prompt = f"""I am trying to solve a math problem, the specific question is: "{question}". \n\n My previous step in solving the problem was as follows:"{history}" \n\n Now I'm reasoning that this step is:"{per_step}" and the specific reasoning is:"{content}". \n\n Please evaluate if my reasoning in this step is correct. \n\n If my current reasoning step is correct and relevant to the question, just answer "yes". \n\n If this step of reasoning is incorrect or irrelevant, you are required to correct it and provide the correct or more relevant step of reasoning directly, surrounded by <<>>."""
    for i in range(10):
        # response = gpt_generate(prompt)
        response = llm_response(prompt)  # 调用生成方法
        # 获取response的第一句话或者如果没有符号就是完整的response
        response_first_sentence = response.split(".")[0]
        # 提取 response 的前三个字符，并将它们转换成小写来进行判断。
        if response_first_sentence[:3].lower() == "yes" or "yes" in response_first_sentence.lower():  
            return 1, response
        else:
            # 尝试修正表达式
            match = re.search(r'<<(.+?)>>', response)
            if match:
                return 0, match.group(1)
    return 0, "Error: LLM cannot generate correct reasoning."

def JudgmentStepReasoningCorrectly(data):
    if 'solution' in data:
        # 获取历史信息
        history = ""
        question = data['question']
        for step, info in data['solution'].items(): # step变量会接收步骤的名称（如"Step 1"），而info变量会接收与这个步骤名称对应的字典值。
            history_json = info['history_json']
            if info['is_calculation_or_reasoning'] == 1: # 如果是计算步
                info['JudgmentStepReasoningCorrectly'], info['StepReasoningCorrectlyResult'] = 1, "This is a calculation step."
            else:
                # 判断并添加新键
                info['JudgmentStepReasoningCorrectly'], info['StepReasoningCorrectlyResult'] = check_reasoning(step, info['content'], question, history)
                # 添加到历史信息中
                if info['JudgmentStepReasoningCorrectly'] == 1:
                    history_json[step] = info['content']
                else:
                    history_json[step] = info['StepReasoningCorrectlyResult']
            history += f"{step}: {history_json[step]}\n"

    # 将处理后的数据转换为字符串以便写入文件
    return data

def postprocess(data):
    # 只保留需要的部分
    need_result = {}
    need_result["question"] = data["question"]

    need_result["response"] = data["response"]
    need_result["answer"] = data["answer"]
    need_result["critic_result"] = data["critic_result"]
    need_result["generated_paths"] = data["generated_paths"]
    need_result["critic_result"] = data["critic_result"]

    need_result['solution'] = {}
    temp_history = {}
    for step, info in data["solution"].items():
        need_result['solution'][step] = {
            "content": info["content"],
            'is_calculation_or_reasoning': info['is_calculation_or_reasoning'],
            "JudgmentStepCalculatedCorrectly": info["JudgmentStepCalculatedCorrectly"],
            "JudgmentStepEquationCorrectly": info["JudgmentStepEquationCorrectly"],
            "JudgmentStepReasoningCorrectly": info["JudgmentStepReasoningCorrectly"],
            "StepCalculatedCorrectlyResult": info["StepCalculatedCorrectlyResult"],
            "StepEquationCorrectlyFormat": info["StepEquationCorrectlyFormat"],
            "StepReasoningCorrectlyResult": info["StepReasoningCorrectlyResult"]
        }
        temp_history = info["history_json"]
    need_result['modifiedResult'] = temp_history
    
    return need_result


def api_both(question, response = None, answer = None):
    data = data = {"question": question}
    
    # 如果提供了回答，就用回答作为回答, 否则生成回答
    if response:
        data["response"] = response
    else:
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
    data_path_pred = generate_process(
        data_back,
        prompt_key = "question",
        response_key = "response"
    )
    
    # 前向过程路径评估
    data_path_pred_judge = evaluate_process(
        data_path_pred,
        backbone = "chatglm_platform",
        prompt_key = "question",
        process_response_key = "generated_paths",
        reference_answewr_key = "answer",
        PROMPT_TEMPLATE = PROMPT_TEMPLATE
    )

    data_path_pred_judge_aggregate = select_math_data_by_rating(
        data_path_pred_judge
    )

    data0 = data_path_pred_judge_aggregate.copy()
    data0["solution"] = data0["response"]
    data1 = SplitByRow(data0) # 数据分行与公式高亮
    data2 = IsCalculationOrReasoning(data1) # 判断计算步还是推理步
    data3 = JudgmentStepCalculatedCorrectly(data2) # 针对计算步的自动标注
    data4 = JudgmentStepReasoningCorrectly(data3) # 针对推理步的自动标注
    out_data = data_path_pred_judge_aggregate
    out_data["solution"] = data4["solution"]

    # 如果全量输出，则关闭下面一行，否则只输出必要信息
    # out_data = postprocess(out_data)

    # 如果有到导出文件的必要，可打开下面一行
    out_to_file(out_data)
    
    return out_data

def main():
    question = "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?"
    result = api_both(question)
    result = json.dumps(result, indent=4, ensure_ascii=False)
    print(result)

if __name__ == '__main__':
    main()