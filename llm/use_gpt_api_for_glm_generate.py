# 适用版本 openai <= 0.28.1

import json
import openai
from tqdm import tqdm
import os
import llm.config as config

# 设定API密钥和基本URL
openai.api_key = config.GPT_API_KEY
openai.api_base = config.GPT_BASE_URL

def gpt_generate(prompt):
    # 构造messages
    messages = [{"role": "user", "content": prompt}]

    # 调用GPT接口
    # model = "gpt-3.5-turbo"
    model = "gpt-4-1106-preview"
    chat_completion = openai.ChatCompletion.create(model=model, messages = messages)
    gpt_response = chat_completion.choices[0].message.content
    return gpt_response

def main():
    prompt = input("请输入问题：")
    print(gpt_generate(prompt))

if __name__ == "__main__":
    main()

