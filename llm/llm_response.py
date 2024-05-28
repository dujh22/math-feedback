import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# import hunter # 用于调试
# hunter.trace(module=__name__)

import time
import json  # 导入json模块，用于处理JSON数据格式
import random  # 导入random模块，用于生成随机数
import requests  # 导入requests模块，用于HTTP请求
import openai  # 导入openai模块，用于调用OpenAI的API

from llm.use_gpt_api_for_glm_generate import gpt_generate
from llm.chatglm import ChatGLM
from llm.config import CRITIC_URL # 从config.py中导入CRITIC_URL
from llm.config import TGI_URL # Import TGI_URL from config.py

# from use_gpt_api_for_glm_generate import gpt_generate
# from chatglm import ChatGLM
# from config import CRITIC_URL # 从config.py中导入CRITIC_URL
# from config import TGI_URL # Import TGI_URL from config.py
ChatGLM = ChatGLM()

TEMPERATURE = 0.9  # 设置生成文本时的温度参数
TOPP = 0.2  # 设置生成文本时的Top-p参数

def query_chatglm_platform(prompt, history=[], do_sample=True, max_tokens=2048, url = CRITIC_URL):
    '''
        该功能用于与基于聊天的模型平台通信，向其发送当前和历史对话数据，然后接收生成的回复。它可以通过温度和top_p 等参数对生成过程进行详细定制，因此适用于需要根据用户输入和以前的对话上下文动态生成回复的交互式应用。
    '''
    # url = "http://xxx:9090/v1/chat/completions"  # 设置API的URL
    # url = CRITIC_URL  # 设置API的URL

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

# 这里设置使用的llm进行生成，注意在本项目中只有这里一个地方进行相关设置
@time_it
def llm_response(prompt, backbone, url):
    # print(prompt[:10])
    response = ""
    # return response
    for i in range(10):
        if backbone == 'tgi':
            try:
                response = query_chatglm_tgi(prompt=prompt, url=url)
                return response
            except:
                continue
        elif backbone == 'chatglm_platform':
            try:
                response = query_chatglm_platform(prompt=prompt, url=url)
                return response
            except:
                continue
        elif backbone == 'glm':
            try:
                response = ChatGLM.generate(prompt)
                return response
            except:
                continue
        else:
            print("采用了未知的backbone", backbone)
            time.sleep(1000)
    return response

# 这里设置使用的llm进行生成，注意在本项目中只有这里一个地方进行相关设置
def llm_response2(prompt, use_glm_or_gpt = 'glm'):
    response = ""
    for i in range(10):
        if use_glm_or_gpt == 'glm':
            try:
                response = ChatGLM.generate(prompt)
                return response
            except:
                continue
        else:
            try:
                response = gpt_generate(prompt)
                return response
            except:
                continue
    return response

USE_GLM_OR_GPT = 'glm'

def llm_response3(prompt, use_glm_or_gpt = USE_GLM_OR_GPT):
    response = ""
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


def main():
    user = input("Q：")
    print("A_tgi: ", llm_response(user, 'tgi', TGI_URL))
    # print("A_critic: ", llm_response(user, 'tgi', CRITIC_URL))
    print("A_glm: ", llm_response(user, 'glm', CRITIC_URL))

if __name__ == "__main__":
    while True:
        main()