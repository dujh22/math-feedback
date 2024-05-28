import json
from openai import OpenAI
from tqdm import tqdm
import os
import llm.config as config

# GLM API密钥和基本URL
api_key = config.GLM_API_KEY
base_url = config.GLM_BASE_URL

# 初始化GLM客户端
client = OpenAI(api_key=api_key, base_url=base_url)

def glm_generate(prompt):
    # 构造messages
    messages = [{"role": "user", "content": prompt}]
    # 调用GLM接口
    stream = client.chat.completions.create(messages=messages, model="chatglm3-32b-v0.8-data", temperature=0.95, top_p=0.7, stream=True, max_tokens=1024)
    response = ''.join(part.choices[0].delta.content or "" for part in stream)

    return response

def main():
    prompt = input("请输入问题：")
    print(glm_generate(prompt))

if __name__ == "__main__":
    main()
