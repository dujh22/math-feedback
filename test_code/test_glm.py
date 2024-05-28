# 适用版本 openai == 1.14.3
from openai import OpenAI
import llm.config as config

client = OpenAI(api_key=config.GLM_API_KEY, base_url=config.GLM_BASE_URL)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "你好！请问你是？",
        }
    ],
    # model="chatglm3-32b-v0.8-data",
    
    model = "glm-4-public",
    temperature=0.95,
    top_p=0.7,
    stream=True,
    max_tokens=1024
)

for part in stream:
    print(part.choices[0].delta.content or "", end="", flush=True)
